"""
TPU-compatible Mamba language model — self-contained.

Differs from `mamba_llm.py`:
  * No `mamba_ssm.utils.generation.GenerationMixin` (training-only model).
  * No `mamba_ssm.utils.hf` (no HuggingFace integration).
  * No relative import of `mamba_block` — RMSNorm / MambaBlockTPU /
    ResidualBlockTPU are vendored inline so this module is the single
    source of truth on TPU.
  * Uses `xla_fused_scan.fused_ssm` (pure-tensor parallel scan) instead of
    the CUDA Triton kernel.
  * `MambaBlockTPU.step()` is fully functional — no `[..., -1] = x`
    in-place writes, which trip up XLA's lazy graph.
  * Optional gradient checkpointing per ResidualBlock with the lambda-
    wrapper trick from `mamba/xla_tpu_reference.md` §4.3 / §7.4.
  * `vocab_size` pads to a multiple of 128 by default (MXU alignment).

Math / shapes are identical to the existing model so checkpoints from
either path are interchangeable (provided the same config is used).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .xla_fused_scan import fused_ssm

# Lazy XLA-aware checkpoint helper with CPU/CUDA fallback.
try:
    from torch_xla.utils.checkpoint import checkpoint as _xla_checkpoint
except ImportError:
    from torch.utils.checkpoint import checkpoint as _xla_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-mean-square normalisation (no centring)."""

    def __init__(self, d_input: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_input))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the rsqrt in fp32 then cast back, per xla_tpu_reference §3.5.
        x32 = x.float()
        norm = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * norm).to(x.dtype) * self.weight


class MambaBlockTPU(nn.Module):
    """Selective SSM block — TPU-friendly."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Linear projections in/out of the inner expansion.
        self.input_proj  = nn.Linear(config.d_input, config.d_model, bias=config.bias)
        self.res_proj    = nn.Linear(config.d_input, config.d_model, bias=config.bias)
        self.output_proj = nn.Linear(config.d_model, config.d_input, bias=config.bias)

        # Depthwise causal convolution.
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.kernel_size,
            padding=config.kernel_size - 1,
            bias=config.conv_bias,
            groups=config.d_model,
        )

        # SSM state-decay matrix A (D, N), HiPPO-style negative diagonal.
        A = repeat(torch.arange(1, config.d_state + 1), "n -> d n", d=config.d_model)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(config.d_model))

        # Selective (input-dependent) SSM parameters.
        self.x_B_proj  = nn.Linear(config.d_model, config.d_state, bias=False)
        self.x_C_proj  = nn.Linear(config.d_model, config.d_state, bias=False)
        self.x_dt_proj = nn.Linear(config.d_model, config.dt_rank, bias=False)
        self.dt_proj   = nn.Linear(config.dt_rank, config.d_model, bias=True)

        # Decode-time cache for A_neg = -exp(A_log.float()). Computed lazily
        # in step() and reused across tokens so the per-step exp+mul kernel
        # launch (~10µs on CUDA, repeated 100+ times per generation) is
        # paid once. Invalidated on train()/eval() and on device move.
        self._A_neg_exp_cache = None

    def train(self, mode: bool = True):
        # Drop the cache so any A_log update during training doesn't
        # silently leak through subsequent eval() calls.
        self._A_neg_exp_cache = None
        return super().train(mode)

    # ── Inference cache helpers ─────────────────────────────────────────────

    def allocate_inference_cache(self, batch_size: int, dtype, device):
        conv_state = torch.zeros(
            batch_size, self.config.d_model, self.config.kernel_size,
            dtype=dtype, device=device,
        )
        ssm_state = torch.zeros(
            batch_size, self.config.d_model, self.config.d_state,
            dtype=torch.float32, device=device,    # state in fp32 for stability
        )
        return conv_state, ssm_state

    # ── SSM (full sequence) ────────────────────────────────────────────────

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        A      = -torch.exp(self.A_log.float())                     # (D, N)
        B_proj = self.x_B_proj(x)                                   # (B, L, N)
        C_proj = self.x_C_proj(x)                                   # (B, L, N)
        delta  = F.softplus(self.dt_proj(self.x_dt_proj(x)))        # (B, L, D)
        return fused_ssm(delta, A, B_proj, x, C_proj, self.D.float())

    # ── Full-sequence forward (training, prefill) ──────────────────────────

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x   = self.input_proj(x_in)
        res = self.res_proj(x_in)
        L   = x.shape[1]

        # Conv path (depthwise causal).
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :L]            # trim causal padding
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.output_proj(y)

    # ── Single-token recurrent step (autoregressive decode) ────────────────

    def step(self, x_in, conv_state, ssm_state):
        """O(1) per-token step — XLA-safe (functional, no in-place writes).

        Args:
            x_in: (B, d_input)  — single token's residual input
            conv_state: (B, d_model, kernel_size)
            ssm_state:  (B, d_model, d_state)  in fp32
        Returns:
            out, new_conv_state, new_ssm_state
        """
        x   = self.input_proj(x_in)             # (B, d_model)
        res = self.res_proj(x_in)               # (B, d_model)

        # Functional buffer roll: shift-left, append new x at end. No in-place.
        conv_state = torch.cat(
            [conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1
        )
        x_conv = (conv_state * self.conv1d.weight[:, 0, :]).sum(-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)                 # (B, d_model)

        # Selective SSM step in fp32 for numerical stability.
        B_proj = self.x_B_proj(x_conv)                            # (B, N)
        C_proj = self.x_C_proj(x_conv)                            # (B, N)
        dt     = F.softplus(self.dt_proj(self.x_dt_proj(x_conv))) # (B, D)
        # A is parameter-stationary at inference; cache the exp+negate
        # so we don't relaunch the kernel per token.
        if (self._A_neg_exp_cache is None
                or self._A_neg_exp_cache.device != self.A_log.device):
            self._A_neg_exp_cache = -torch.exp(self.A_log.float())
        A = self._A_neg_exp_cache                                  # (D, N)

        dA  = torch.exp(torch.clamp(
            dt.float().unsqueeze(-1) * A,                          # (B, D, N)
            min=-20.0, max=0.0,
        ))
        dBu = (dt.float().unsqueeze(-1)
               * B_proj.float().unsqueeze(1)
               * x_conv.float().unsqueeze(-1))                     # (B, D, N)

        ssm_state = dA * ssm_state + dBu                           # (B, D, N)

        y = ((ssm_state.to(x_conv.dtype) * C_proj.unsqueeze(1)).sum(-1)
             + self.D * x_conv)                                    # (B, d_model)
        y = y * F.silu(res)
        return self.output_proj(y), conv_state, ssm_state


class ResidualBlockTPU(nn.Module):
    """RMSNorm → MambaBlockTPU → +residual, with optional checkpointing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mamba_block = MambaBlockTPU(config)
        self.norm = RMSNorm(config.d_input)
        self.use_checkpoint = bool(getattr(config, "use_checkpoint", False))

    def allocate_inference_cache(self, batch_size, dtype, device):
        return self.mamba_block.allocate_inference_cache(batch_size, dtype, device)

    def step(self, x, conv_state, ssm_state):
        out, conv_state, ssm_state = self.mamba_block.step(
            self.norm(x), conv_state, ssm_state
        )
        return out + x, conv_state, ssm_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint and x.requires_grad:
            # Lambda wrapper avoids the inspect.ismethod() barrier explosion
            # described in xla_tpu_reference.md §4.3.
            def _ckpt_fn(x_):
                return self.mamba_block(self.norm(x_))
            return _xla_checkpoint(_ckpt_fn, x, use_reentrant=True) + x
        return self.mamba_block(self.norm(x)) + x


# ─────────────────────────────────────────────────────────────────────────────
# Config & full LM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MambaLMConfig:
    # Tokenizer
    vocab_size:               int  = 65
    pad_vocab_size_multiple:  int  = 128       # MXU alignment

    # Architecture
    n_layer:     int  = 4
    d_input:     int  = 128
    d_model:     int  = 256                     # 2 × d_input
    d_state:     int  = 16
    dt_rank:     int  = 16                      # ≈ d_model / 16
    kernel_size: int  = 4
    bias:        bool = False
    conv_bias:   bool = True

    # Tying & training knobs
    tie_embeddings: bool = True
    use_checkpoint: bool = False
    seq_len:        int  = 512                  # for inference-time decoding

    def __post_init__(self):
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - (self.vocab_size % self.pad_vocab_size_multiple)
            )


class MambaLMHeadModel(nn.Module):
    """Embedding → ResidualBlockTPU × N → RMSNorm → LM head."""

    def __init__(self, config: MambaLMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_input)
        self.layers    = nn.ModuleList([
            ResidualBlockTPU(config) for _ in range(config.n_layer)
        ])
        self.norm_f  = RMSNorm(config.d_input)
        self.lm_head = nn.Linear(config.d_input, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # GPT-2 style init (matches the existing mamba_llm.py).
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module, initializer_range: float = 0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

    # ── Training / prefill forward ──────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) integer; returns logits (B, L, V)."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    # ── O(1) decode ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def allocate_inference_cache(self, batch_size: int,
                                 dtype: torch.dtype = torch.float32,
                                 device=None):
        return [
            layer.allocate_inference_cache(batch_size, dtype, device)
            for layer in self.layers
        ]

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, caches):
        """One-token step.

        Args:
            input_ids: (B,) ints — single next token
            caches:    list of (conv_state, ssm_state), one per layer
        Returns:
            logits: (B, V)
            new_caches: list[(conv_state, ssm_state)]
        """
        x = self.embedding(input_ids)               # (B, d_input)
        new_caches = []
        for layer, (cs, ss) in zip(self.layers, caches):
            x, cs, ss = layer.step(x, cs, ss)
            new_caches.append((cs, ss))
        x = self.norm_f(x)
        return self.lm_head(x), new_caches

    # ── Convenience parameter accounting ────────────────────────────────────

    def num_parameters(self, unique: bool = True) -> int:
        """Count parameters.

        `unique=True` (default) counts each Parameter object once — this is
        what `model.parameters()` returns natively (tied weights dedupe
        automatically). This is the standard convention used in HF model
        cards (e.g. GPT-2 124M).

        `unique=False` counts a tied embedding twice — the "all storage"
        view, occasionally reported but rarely meaningful.
        """
        n = sum(p.numel() for p in self.parameters())
        if not unique and self.config.tie_embeddings:
            n += self.embedding.weight.numel()   # add the tied copy back
        return n
