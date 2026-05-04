"""
Mamba SSM forward optimised for PyTorch/XLA on Google Cloud TPU v4.

Drop-in replacement for `mamba.fused_scan.fused_ssm` on XLA devices.
The math is identical to `fused_scan._fused_ssm_ref` (the parity oracle for tests),
but the time-axis recurrence is solved with a parallel associative scan instead
of a Python `for t in range(L)` loop — the latter would compile a graph
proportional to L and blow up XLA trace time.

Two scan variants are provided:

    "hillis"  — Hillis-Steele O(L log L) work, O(log L) depth.
                Best for L <= 1K.

    "chunked" — Sequential within K-element chunks (XLA unrolls), then
                Hillis-Steele across chunks.
                O(L log(L/K)) work, O(K + log(L/K)) depth.
                Best for L >= 2K.

    "auto"    — chunked if L > 2*chunk_size, else hillis.

Backward is handled by PyTorch autograd through pure-tensor ops.
Activation-memory pressure should be handled at the block level via
`torch_xla.utils.checkpoint` on `ResidualBlockTPU.forward` (already wired
in `mamba/mamba_llm_tpu.py`). A scan-level checkpoint wrapper would be
redundant — the outer block checkpoint already recomputes the scan in
backward.
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CLAMP_LO = -20.0           # exp(-20) ~ 2e-9; safe lower bound for fp32
_CLAMP_HI = 0.0             # A is negative so delta*A <= 0 always
_MIN_L_POW2 = 32
_COMPUTE_DTYPE = torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= max(n, _MIN_L_POW2)."""
    if n <= _MIN_L_POW2:
        return _MIN_L_POW2
    return 1 << (n - 1).bit_length()


def _pad_to_pow2(t: torch.Tensor, L_target: int, dim: int, fill_value: float) -> torch.Tensor:
    """Right-pad `t` along `dim` from its current length to `L_target` with `fill_value`."""
    cur = t.shape[dim]
    if cur == L_target:
        return t
    n_dims_after = t.dim() - dim - 1
    pad = []
    for _ in range(n_dims_after):
        pad.extend([0, 0])
    pad.extend([0, L_target - cur])
    return F.pad(t, pad, value=fill_value)


def _hillis_steele_scan(gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Inclusive parallel scan with combine `((g1,v1),(g2,v2)) -> (g1*g2, v1*g2 + v2)`.

    Args:
        gates, tokens: (..., L) where L is a power of 2.
    Returns:
        h: (..., L) where h[t] = gates[t]*h[t-1] + tokens[t], h[-1] = 0.
    """
    L = gates.shape[-1]
    assert L > 0 and (L & (L - 1)) == 0, f"L must be a power of 2, got {L}"
    K = (L - 1).bit_length()                       # = log2(L), Python int

    g_cur = gates
    v_cur = tokens
    for k in range(K):                              # XLA unrolls this loop
        d = 1 << k
        # Left-shift by d, padding the leading d slots with identity (g=1, v=0).
        g_prev = F.pad(g_cur[..., :L - d], (d, 0), value=1.0)
        v_prev = F.pad(v_cur[..., :L - d], (d, 0), value=0.0)
        # combine(prev, cur): apply prev FIRST, then cur.
        # NOTE: read v_prev * g_cur BEFORE overwriting g_cur.
        new_g = g_prev * g_cur
        new_v = v_prev * g_cur + v_cur
        g_cur, v_cur = new_g, new_v

    return v_cur


def _chunked_scan(gates: torch.Tensor, tokens: torch.Tensor, K: int) -> torch.Tensor:
    """Chunked parallel scan: sequential within K-blocks, parallel across blocks.

    Args:
        gates, tokens: (..., L) where L % K == 0 and L/K is a power of 2.
        K: chunk size (compile-time constant).
    Returns:
        h: (..., L) — same as `_hillis_steele_scan` but with reduced work.
    """
    prefix = gates.shape[:-1]
    L = gates.shape[-1]
    assert L % K == 0, f"L={L} must be divisible by K={K}"
    M = L // K
    assert M > 0 and (M & (M - 1)) == 0, f"L/K = {M} must be a power of 2"

    # 1. Reshape into (..., M, K).
    g = gates.view(*prefix, M, K)
    v = tokens.view(*prefix, M, K)

    # 2. Within-chunk inclusive scan, sequential. Stack the K running values
    #    instead of in-place index_copy — both unroll identically under XLA,
    #    but stack is cleaner and obviously functional.
    g_running_list = []
    v_running_list = []
    g_running = torch.ones_like(g[..., 0])     # identity gate
    v_running = torch.zeros_like(v[..., 0])    # zero starting state per chunk
    for k in range(K):                          # static K → XLA unrolls
        g_k = g[..., k]
        v_k = v[..., k]
        g_running = g_running * g_k
        v_running = v_running * g_k + v_k
        g_running_list.append(g_running)
        v_running_list.append(v_running)
    g_intra = torch.stack(g_running_list, dim=-1)   # (..., M, K) cumulative gate
    v_intra = torch.stack(v_running_list, dim=-1)   # (..., M, K) inclusive h, carry-zero

    # 3. Per-chunk endpoint pair (g_chunk_total, v_chunk_carry_zero).
    g_chunk_end = g_intra[..., -1]                  # (..., M)
    v_chunk_end = v_intra[..., -1]                  # (..., M)

    # 4. Inclusive scan across chunks → exclusive carry by right-shifting one slot.
    v_inc = _hillis_steele_scan(g_chunk_end, v_chunk_end)
    carry_v = F.pad(v_inc[..., :-1], (1, 0), value=0.0)  # (..., M)

    # 5. Broadcast carry into each position of its chunk:
    #    h[..., m, k] = v_intra[..., m, k] + g_intra[..., m, k] * carry_v[..., m]
    h = v_intra + g_intra * carry_v.unsqueeze(-1)        # (..., M, K)

    # 6. Reshape back.
    return h.view(*prefix, L)


def _discretize(delta: torch.Tensor,
                A: torch.Tensor,
                B_proj: torch.Tensor,
                x: torch.Tensor) -> tuple:
    """Compute (gates, tokens) of shape (B, L, D, N), all in fp32.

    gates[b,l,d,n]  = exp( clamp(delta[b,l,d] * A[d,n], -20, 0) )
    tokens[b,l,d,n] = delta[b,l,d] * B_proj[b,l,n] * x[b,l,d]
    """
    arg = delta.unsqueeze(-1) * A                                  # (B,L,D,N)
    arg = torch.clamp(arg, min=_CLAMP_LO, max=_CLAMP_HI)
    gates = torch.exp(arg)
    tokens = (delta.unsqueeze(-1)
              * B_proj.unsqueeze(2)
              * x.unsqueeze(-1))                                   # (B,L,D,N)
    return gates, tokens


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fused_ssm(delta: torch.Tensor,
              A: torch.Tensor,
              B_proj: torch.Tensor,
              x: torch.Tensor,
              C_proj: torch.Tensor,
              D_param: torch.Tensor,
              *,
              scan: str = "auto",
              chunk_size: int = 128) -> torch.Tensor:
    """XLA/TPU-optimised Mamba SSM forward.

    Args:
        delta:   (B, L, D)  timestep sizes (after softplus)
        A:       (D, N)     state decay matrix (negative; see mamba_block.py:81)
        B_proj:  (B, L, N)  input projection
        x:       (B, L, D)  input signal
        C_proj:  (B, L, N)  output projection
        D_param: (D,)       feedthrough
        scan:    "hillis" | "chunked" | "auto" (default "auto")
        chunk_size: chunk length used by the chunked scan (default 128)

    Returns:
        y: (B, L, D), same dtype as `x`.

    Notes:
        Internally promotes all inputs to fp32 for numerical stability of
        the cumulative gate product, then casts the output back to x.dtype.
    """
    out_dtype = x.dtype

    # 1. Promote to fp32.
    delta_f   = delta.to(_COMPUTE_DTYPE)
    A_f       = A.to(_COMPUTE_DTYPE)
    B_proj_f  = B_proj.to(_COMPUTE_DTYPE)
    x_f       = x.to(_COMPUTE_DTYPE)
    C_proj_f  = C_proj.to(_COMPUTE_DTYPE)
    D_param_f = D_param.to(_COMPUTE_DTYPE)

    B, L, D = delta_f.shape
    # N = A_f.shape[1]   # not directly needed below

    # 2. Discretize → (B, L, D, N).
    gates, tokens = _discretize(delta_f, A_f, B_proj_f, x_f)

    # 3. Pad L to a power of 2 (no-op when L is already pow2).
    L_pad = _next_pow2(L)
    if L_pad != L:
        gates  = _pad_to_pow2(gates,  L_pad, dim=1, fill_value=1.0)
        tokens = _pad_to_pow2(tokens, L_pad, dim=1, fill_value=0.0)

    # 4. Move time to last dim → (B, D, N, L_pad). One transpose; cheap.
    g_t = gates.permute(0, 2, 3, 1).contiguous()
    v_t = tokens.permute(0, 2, 3, 1).contiguous()

    # 5. Run the scan.
    if scan == "auto":
        scan = "chunked" if L_pad > 2 * chunk_size else "hillis"

    if scan == "hillis":
        h_t = _hillis_steele_scan(g_t, v_t)
    elif scan == "chunked":
        K = min(chunk_size, L_pad)
        # Ensure L_pad / K is a power of 2 (so the across-chunks Hillis-Steele works).
        if (L_pad // K) & ((L_pad // K) - 1) != 0:
            raise ValueError(
                f"chunk_size={chunk_size} does not yield a power-of-2 chunk count "
                f"for padded L={L_pad}. Choose K such that L_pad/K is a power of 2."
            )
        if L_pad % K != 0:
            raise ValueError(
                f"chunk_size={chunk_size} does not divide padded L={L_pad}."
            )
        h_t = _chunked_scan(g_t, v_t, K)
    else:
        raise ValueError(f"unknown scan={scan!r}; expected 'hillis', 'chunked', or 'auto'")

    # 6. Trim padding, restore (B, L, D, N) layout.
    h = h_t[..., :L].permute(0, 3, 1, 2).contiguous()             # (B, L, D, N)

    # 7. Output projection: y = einsum('bldn,bln->bld', h, C) + D * x.
    y = torch.einsum('bldn,bln->bld', h, C_proj_f) + D_param_f * x_f

    return y.to(out_dtype)
