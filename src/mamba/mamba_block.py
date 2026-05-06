"""Reference Mamba block — selective state-space model + depthwise conv + gating.

Implements the canonical Mamba layer in plain PyTorch. ``MambaBlock`` exposes
both a parallel ``forward`` (for training and prefill) and an O(1) recurrent
``step`` (for autoregressive decode). ``ResidualBlock`` wraps a block with
RMSNorm + residual connection so it can be stacked into the LM in
:mod:`mamba.mamba_llm`.

The TPU/XLA and CUDA/Triton variants in :mod:`mamba.mamba_llm_tpu` and
:mod:`mamba.mamba_llm_cuda` keep the same parameter shapes and ``state_dict``
keys as this reference implementation, so the same checkpoint loads into any
of them with ``strict=True``.

Sources:
- https://arxiv.org/abs/2312.00752     (Mamba: Selective SSMs, Gu & Dao 2023)
- https://www.ibm.com/think/topics/mamba-model
- https://arxiv.org/pdf/2111.00396     (S4)
- https://arxiv.org/pdf/2008.07669     (HiPPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .fused_scan import fused_ssm


class MambaBlock(nn.Module):
    """A single Mamba layer.

    Forward pass:
        1. Linear projections on input → ``x`` and ``res``.
        2. Depthwise causal 1-D convolution on ``x``.
        3. SiLU activation.
        4. Selective SSM (input-dependent ``Δ``, ``B``, ``C``).
        5. Gate by ``SiLU(res)``.
        6. Output projection back to ``d_input``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.d_input, config.d_model, bias=config.bias)
        self.res_proj = nn.Linear(config.d_input, config.d_model, bias=config.bias)
        self.output_proj = nn.Linear(config.d_model, config.d_input, bias=config.bias)

        # Depthwise causal conv: groups == in_channels, padding picks up the
        # final L tokens (left padding is trimmed in forward).
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.kernel_size,
            padding=config.kernel_size - 1,
            bias=config.conv_bias,
            groups=config.d_model,
        )

        # State transition matrix A is parameterised as log(A) with A
        # initialised to a diagonal HiPPO-style spectrum (1..d_state). At
        # call time we use -exp(A_log) so A stays strictly negative, keeping
        # the discrete-time recurrence a contraction.
        A = repeat(torch.arange(1, config.d_state + 1), "n -> d n", d=config.d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_model))

        # Input-dependent projections producing the selective B, C, and Δ.
        self.x_B_proj = nn.Linear(config.d_model, config.d_state, bias=False)
        self.x_C_proj = nn.Linear(config.d_model, config.d_state, bias=False)
        self.x_dt_proj = nn.Linear(config.d_model, config.dt_rank, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_model, bias=True)

        # Decode-time cache for ``A_neg = -exp(A_log.float())``. Computed
        # lazily on the first ``step()`` call so the per-token kernel launch
        # (~10 µs) is amortised. Invalidated on train()/eval() and on a
        # device move.
        self._A_neg_exp_cache = None

    def train(self, mode: bool = True):
        self._A_neg_exp_cache = None
        return super().train(mode)

    def allocate_inference_cache(self, batch_size, dtype, device):
        """Allocate the per-batch (conv_state, ssm_state) tuple used by ``step``."""
        conv_state = torch.zeros(
            batch_size, self.config.d_model, self.config.kernel_size,
            dtype=dtype, device=device,
        )
        ssm_state = torch.zeros(
            batch_size, self.config.d_model, self.config.d_state,
            dtype=torch.float32, device=device,
        )
        return conv_state, ssm_state

    def ssm(self, x):
        """Selective SSM over a full sequence.

        Args:
            x: ``(B, L, d_model)`` post-conv, post-activation sequence.

        Returns:
            ``(B, L, d_model)`` SSM output.
        """
        A = -torch.exp(self.A_log.float())                       # (D, N)
        B_proj = self.x_B_proj(x)                                # (B, L, N)
        C_proj = self.x_C_proj(x)                                # (B, L, N)
        delta = F.softplus(self.dt_proj(self.x_dt_proj(x)))      # (B, L, D)
        return fused_ssm(delta, A, B_proj, x, C_proj, self.D.float())

    def step(self, x_in, conv_state, ssm_state):
        """Single-token recurrent step for O(1) autoregressive inference.

        Args:
            x_in:       ``(B, d_input)`` next-token activation.
            conv_state: ``(B, d_model, kernel_size)`` sliding conv buffer.
            ssm_state:  ``(B, d_model, d_state)`` current SSM hidden state (fp32).

        Returns:
            Tuple ``(output, conv_state, ssm_state)`` with ``output`` of shape
            ``(B, d_input)`` and the updated state buffers.
        """
        x = self.input_proj(x_in)
        res = self.res_proj(x_in)

        # Causal conv: shift buffer left, insert new token at the end.
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x
        x_conv = (conv_state * self.conv1d.weight[:, 0, :]).sum(-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)

        B_proj = self.x_B_proj(x_conv)
        C_proj = self.x_C_proj(x_conv)
        dt = F.softplus(self.dt_proj(self.x_dt_proj(x_conv)))

        # Discretise + SSM update in fp32 for numerical stability.
        if (self._A_neg_exp_cache is None
                or self._A_neg_exp_cache.device != self.A_log.device):
            self._A_neg_exp_cache = -torch.exp(self.A_log.float())
        A = self._A_neg_exp_cache
        dA = torch.exp(dt.float().unsqueeze(-1) * A)
        dBu = (dt.float().unsqueeze(-1)
               * B_proj.float().unsqueeze(1)
               * x_conv.float().unsqueeze(-1))
        ssm_state = dA * ssm_state + dBu

        y = ((ssm_state.to(x_conv.dtype) * C_proj.unsqueeze(1)).sum(-1)
             + self.D * x_conv)
        y = y * F.silu(res)
        return self.output_proj(y), conv_state, ssm_state

    def forward(self, x_in):
        """Parallel forward over a full sequence ``(B, L, d_input)``."""
        x = self.input_proj(x_in)
        res = self.res_proj(x_in)
        L = x.shape[1]

        # Depthwise causal conv: trim the right-padding back to L.
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.output_proj(y)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019)."""

    def __init__(self, d_input, eps=1e-5):
        super().__init__()
        self.d_model = d_input
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_input))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ResidualBlock(nn.Module):
    """RMSNorm + ``MambaBlock`` + residual; the unit stacked inside ``MambaLLM``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mamba_block = MambaBlock(config)
        self.norm = RMSNorm(config.d_input)

    def allocate_inference_cache(self, batch_size, dtype, device):
        return self.mamba_block.allocate_inference_cache(batch_size, dtype, device)

    def step(self, x, conv_state, ssm_state):
        out, conv_state, ssm_state = self.mamba_block.step(
            self.norm(x), conv_state, ssm_state
        )
        return out + x, conv_state, ssm_state

    def forward(self, x):
        return self.mamba_block(self.norm(x)) + x
