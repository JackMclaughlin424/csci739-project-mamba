"""
CUDA-targeted Mamba language model — Triton-backed inference path.

Drop-in replacement for `mamba.mamba_llm_tpu.MambaLMHeadModel` whose only
purpose is to make Colab/CUDA inference fast. Differences vs the TPU
module:

  * `MambaBlock` and `ResidualBlock` come from `mamba.mamba_block` — the
    SSM dispatches to the Triton kernel in `mamba.fused_scan` when CUDA
    is present, falling back to the (slow) sequential reference scan
    only on CPU. The TPU module's pure-PyTorch parallel-scan is never
    used here.
  * `step()` uses `mamba_block.MambaBlock.step` which keeps the conv-state
    update in-place via `torch.roll` — saves an allocation per layer per
    decoded token vs the XLA-safe functional `cat`.
  * No gradient-checkpointing branch — this module is inference-only.
  * Re-exports the same `MambaLMConfig` dataclass so checkpoints saved by
    `mamba_llm_tpu` load with `strict=True` against this model. Parameter
    names match exactly: `embedding.weight`, `layers.{i}.mamba_block.*`,
    `layers.{i}.norm.*`, `norm_f.*`, `lm_head.weight`.

Recommended usage (from a Colab cell):

    from mamba.mamba_llm_cuda import MambaLMHeadModel, MambaLMConfig
    cfg = MambaLMConfig(**ckpt["config"])
    model = MambaLMHeadModel(cfg).cuda().eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)

If Triton is unavailable (older Colab images, CPU-only sessions), the
import still succeeds and `fused_ssm` falls back to `_fused_ssm_ref` —
correct but unusably slow at L >= 256. Guard with `HAS_TRITON` if that
matters.
"""

import torch
import torch.nn as nn

from .mamba_block import MambaBlock, RMSNorm, ResidualBlock
from .mamba_llm_tpu import MambaLMConfig

try:
    from .fused_scan import HAS_TRITON
except ImportError:
    HAS_TRITON = False


__all__ = ["MambaLMHeadModel", "MambaLMConfig", "HAS_TRITON"]


class MambaLMHeadModel(nn.Module):
    """Embedding → ResidualBlock × N → RMSNorm → LM head (CUDA/Triton)."""

    def __init__(self, config: MambaLMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_input)
        self.layers    = nn.ModuleList([
            ResidualBlock(config) for _ in range(config.n_layer)
        ])
        self.norm_f  = RMSNorm(config.d_input)
        self.lm_head = nn.Linear(config.d_input, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Match the TPU module's GPT-2 style init so a fresh-from-config
        # build (rare; the typical use is load_state_dict) produces the
        # same initial parameters.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module, initializer_range: float = 0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) int → logits (B, L, V)."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def allocate_inference_cache(self, batch_size: int,
                                 dtype: torch.dtype = torch.float32,
                                 device=None):
        return [
            layer.allocate_inference_cache(batch_size, dtype, device)
            for layer in self.layers
        ]

    @torch.inference_mode()
    def step(self, input_ids: torch.Tensor, caches):
        """One-token step.

        Args:
            input_ids: (B,) int — single next token id
            caches: list of (conv_state, ssm_state) pairs, one per layer
        Returns:
            logits: (B, V), new_caches: list[(conv_state, ssm_state)]
        """
        x = self.embedding(input_ids)
        new_caches = []
        for layer, (cs, ss) in zip(self.layers, caches):
            x, cs, ss = layer.step(x, cs, ss)
            new_caches.append((cs, ss))
        x = self.norm_f(x)
        return self.lm_head(x), new_caches

    def num_parameters(self, unique: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not unique and self.config.tie_embeddings:
            n += self.embedding.weight.numel()
        return n
