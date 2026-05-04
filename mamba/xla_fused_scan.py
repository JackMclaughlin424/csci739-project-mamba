"""
Mamba SSM forward optimised for PyTorch/XLA on Google Cloud TPU v4.

Drop-in replacement for `mamba.fused_scan.fused_ssm` on XLA devices.
The math is identical to `fused_scan._fused_ssm_ref` (the parity oracle for tests),
but the time-axis recurrence is solved with a parallel associative scan instead
of a Python `for t in range(L)` loop — the latter would compile a graph
proportional to L and blow up XLA trace time.

Single public scan path (post-Phase-4): chunked, with chunk size K
chosen per L from `K_TABLE` to minimise the work-count cost model
`K + (L/K)·log2(L/K)`. Internally this path uses Hillis-Steele for the
across-chunks parallel reduction (`_hillis_steele_scan`); both helpers
operate on (B, L, D, N) with time on dim=1, no layout permutes.

The `scan` argument is kept for backward compatibility:

    "auto"    — chunked with K from `K_TABLE` (default).
    "hillis"  — alias for "auto" (legacy; previously selected the
                Hillis-Steele algorithm directly, now equivalent).
    "chunked" — chunked with the explicit `chunk_size` argument.

Backward is handled by PyTorch autograd through pure-tensor ops.
Activation-memory pressure should be handled at the block level via
`torch_xla.utils.checkpoint` on `ResidualBlockTPU.forward` (already wired
in `mamba/mamba_llm_tpu.py`). A scan-level checkpoint wrapper would be
redundant — the outer block checkpoint already recomputes the scan in
backward.
"""

import warnings

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CLAMP_LO = -20.0           # exp(-20) ~ 2e-9; safe lower bound for fp32
_CLAMP_HI = 0.0             # A is negative so delta*A <= 0 always
_MIN_L_POW2 = 32
_COMPUTE_DTYPE = torch.float32

# Per-L chunk-size lookup. K minimises the work-count cost model
# `within_chunk(K) + across_chunk(L/K)` ≈ K + (L/K) * log2(L/K), with the
# constraint that K is a power of 2 dividing L_pad and that L_pad/K is a
# power of 2. Verified by audit (`xla_scan_optimizations.md` §5); requires
# empirical tuning per (L, d_model) on TPU before being treated as final.
K_TABLE: dict = {
    32:    32,   # M=1 (degenerate; equivalent to a pure unrolled loop)
    64:    16,
    128:   32,
    256:   32,
    512:   64,
    1024:  64,
    2048: 128,
    4096: 128,
    8192: 256,
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _einsum_promoted(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """`torch.einsum` with dtype promotion to the highest-precision operand.

    CPU's einsum/matmul refuses mixed dtypes (`bf16 × fp32` errors out),
    but TPU's `dot_general` HLO accepts them via an implicit `convert`
    op fused into the producer. This wrapper hand-codes the same
    promotion so the code path works uniformly on both backends — under
    Phase 5 of `xla_scan_optimizations.md`, several einsum operands
    differ in dtype (e.g. `h_fp32 × C_proj_bf16` for the y projection).
    The cast is a free type op on TPU and a single materialised copy on
    CPU; either way it preserves the original-tensor storage choice.
    """
    common = operands[0].dtype
    for t in operands[1:]:
        common = torch.promote_types(common, t.dtype)
    promoted = tuple(t if t.dtype == common else t.to(common) for t in operands)
    return torch.einsum(equation, *promoted)


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
        gates, tokens: shape (B, L, ...) with TIME ON DIM=1 and L a power
            of 2. The trailing dims are independent feature axes (D, N for
            the main scan; just D for the across-chunks scan inside
            `_chunked_scan`). Operating on `dim=1` directly avoids the
            `permute(0,2,3,1).contiguous()` round-trip that the previous
            time-last layout required (Phase 3 of `xla_scan_optimizations.md`).
    Returns:
        h: same shape as inputs; h[:, t] = gates[:, t]*h[:, t-1] + tokens[:, t].
    """
    L = gates.shape[1]
    assert L > 0 and (L & (L - 1)) == 0, f"L must be a power of 2, got {L}"
    n_iter = (L - 1).bit_length()                  # = log2(L), Python int
    n_trail = gates.dim() - 2                       # feature dims after time

    g_cur = gates
    v_cur = tokens
    for k in range(n_iter):                         # XLA unrolls this loop
        d = 1 << k
        # Left-shift dim=1 by d, padding leading d slots with identity
        # (g=1, v=0). F.pad's spec pairs run from the last dim backwards;
        # zeros for every trailing feature dim, then (d, 0) for time.
        pad = (0, 0) * n_trail + (d, 0)
        g_prev = F.pad(g_cur[:, :L - d], pad, value=1.0)
        v_prev = F.pad(v_cur[:, :L - d], pad, value=0.0)
        # combine(prev, cur): apply prev FIRST, then cur.
        # NOTE: read v_prev * g_cur BEFORE overwriting g_cur.
        new_g = g_prev * g_cur
        new_v = v_prev * g_cur + v_cur
        g_cur, v_cur = new_g, new_v

    return v_cur


def _chunked_scan(gates: torch.Tensor, tokens: torch.Tensor, K: int) -> torch.Tensor:
    """Chunked parallel scan: sequential within K-blocks, parallel across blocks.

    Args:
        gates, tokens: (B, L, D, N) with time on dim=1, L % K == 0, L/K
            a power of 2. (Layout consistent with `_hillis_steele_scan`
            after the Phase 3 refactor.)
        K: chunk size (compile-time constant).
    Returns:
        h: (B, L, D, N).
    """
    B, L, D, N = gates.shape
    assert L % K == 0, f"L={L} must be divisible by K={K}"
    M = L // K
    assert M > 0 and (M & (M - 1)) == 0, f"L/K = {M} must be a power of 2"

    # 1. Reshape: (B, M, K, D, N) — chunks on dim=1, intra-chunk on dim=2.
    g = gates.view(B, M, K, D, N)
    v = tokens.view(B, M, K, D, N)

    # 2. Within-chunk inclusive scan, sequential over dim=2 (XLA unrolls).
    #    The running carry is held in `_COMPUTE_DTYPE` (fp32) so the
    #    cumulative product `g_running *= g_k` cannot underflow at long L
    #    when `g_k` is bf16. PyTorch's type-promotion rule promotes the
    #    bf16 × fp32 multiply to fp32, so we keep the carry-precision
    #    invariant without explicit casts inside the loop. (Phase 5 of
    #    `xla_scan_optimizations.md`; per-chunk g_intra also needs fp32.)
    leading_shape = g.shape[:-3] + g.shape[-2:]     # (..., M, D, N)
    g_running = torch.ones(leading_shape, dtype=_COMPUTE_DTYPE,
                           device=g.device)
    v_running = torch.zeros(leading_shape, dtype=_COMPUTE_DTYPE,
                            device=v.device)
    g_running_list = []
    v_running_list = []
    for k in range(K):                              # static K → XLA unrolls
        g_k = g[:, :, k]
        v_k = v[:, :, k]
        g_running = g_running * g_k
        v_running = v_running * g_k + v_k
        g_running_list.append(g_running)
        v_running_list.append(v_running)
    g_intra = torch.stack(g_running_list, dim=2)    # (B, M, K, D, N) fp32
    v_intra = torch.stack(v_running_list, dim=2)    # (B, M, K, D, N) fp32

    # 3. Per-chunk endpoint pair (g_chunk_total, v_chunk_carry_zero).
    g_chunk_end = g_intra[:, :, -1]                 # (B, M, D, N)
    v_chunk_end = v_intra[:, :, -1]                 # (B, M, D, N)

    # 4. Across-chunks inclusive scan → exclusive carry by right-shifting
    #    one slot along dim=1.
    v_inc = _hillis_steele_scan(g_chunk_end, v_chunk_end)   # (B, M, D, N)
    carry_v = torch.cat([
        torch.zeros_like(v_inc[:, :1]),
        v_inc[:, :-1],
    ], dim=1)                                                # (B, M, D, N)

    # 5. Broadcast carry across positions inside each chunk:
    #    h[:, m, k, d, n] = v_intra[:, m, k, d, n] + g_intra[:, m, k, d, n] * carry_v[:, m, d, n]
    h = v_intra + g_intra * carry_v.unsqueeze(2)             # (B, M, K, D, N)

    # 6. Reshape back to (B, L, D, N).
    return h.view(B, L, D, N)


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


def _resolve_scan_and_K(scan_mode: str, L_pad: int, chunk_size: int) -> tuple:
    """Resolve scan mode + chunk size; everything routes through chunked.

    Phase 4 of `xla_scan_optimizations.md` collapses the user-facing
    `hillis` vs `chunked` dispatch into a single chunked path (one
    compiled HLO topology per (L, K) instead of two). `_hillis_steele_scan`
    is still used INTERNALLY by `_chunked_scan` for the across-chunks
    scan; only the public dispatch is unified.

    - `scan="auto"`   → chunked with K from `K_TABLE` (or chunk_size for
      lengths not in the table).
    - `scan="hillis"` → silent alias for `"auto"` (kept for backward
      compatibility with existing callers).
    - `scan="chunked"` → chunked with the user-supplied `chunk_size`.

    Returns `("chunked", K)` always.
    """
    if scan_mode == "hillis":
        warnings.warn(
            "scan='hillis' is deprecated and now aliases scan='auto'. The "
            "Hillis-Steele algorithm is no longer the public dispatch path "
            "(it is still used internally by the chunked scan for the "
            "across-chunks reduction). If you depend on the original "
            "O(L log L) work / O(log L) depth behaviour, that depth "
            "guarantee is no longer met — see Phase 4 of "
            "`xla_scan_optimizations.md`.",
            DeprecationWarning,
            stacklevel=3,
        )
        K = K_TABLE.get(L_pad, chunk_size)
    elif scan_mode == "auto":
        K = K_TABLE.get(L_pad, chunk_size)
    elif scan_mode == "chunked":
        K = chunk_size
    else:
        raise ValueError(
            f"unknown scan={scan_mode!r}; expected 'hillis', 'chunked', or 'auto'"
        )

    K = min(K, L_pad)
    if L_pad % K != 0:
        raise ValueError(
            f"chunk_size={K} does not divide padded L={L_pad}."
        )
    if (L_pad // K) & ((L_pad // K) - 1) != 0:
        raise ValueError(
            f"chunk_size={K} does not yield a power-of-2 chunk count for "
            f"padded L={L_pad}. Choose K such that L_pad/K is a power of 2."
        )
    return "chunked", K


def _run_scan(gates_blnd: torch.Tensor,
              tokens_blnd: torch.Tensor,
              scan_mode: str,
              chunk_size: int) -> tuple:
    """Pad to pow2, dispatch scan, return h of shape (B, L, D, N).

    Inputs are (B, L, D, N) with time on dim=1 (matches `_discretize`
    output). The inner scan kernels also operate on dim=1, so this
    helper does NO layout permutes — the previous `permute(0,2,3,1).
    contiguous()` round-trip (~6 GB HBM traffic per scan call at 5M
    training shapes) is gone (Phase 3 of `xla_scan_optimizations.md`).

    Returns (h, resolved_scan_mode); the resolved mode is "hillis" or
    "chunked" so callers can re-use it on a second pass (e.g. backward).
    """
    L = gates_blnd.shape[1]
    L_pad = _next_pow2(L)
    if L_pad != L:
        gates_blnd  = _pad_to_pow2(gates_blnd,  L_pad, dim=1, fill_value=1.0)
        tokens_blnd = _pad_to_pow2(tokens_blnd, L_pad, dim=1, fill_value=0.0)

    scan_mode, K = _resolve_scan_and_K(scan_mode, L_pad, chunk_size)
    # Single dispatch path post-Phase-4: chunked handles the M=1 edge
    # case (degenerates to a sequential within-chunk loop with a trivial
    # across-chunks scan), so it covers everything that used to be
    # routed to `_hillis_steele_scan` from the public API.
    h_pad = _chunked_scan(gates_blnd, tokens_blnd, K)

    # Trim padding (no-op when L is already a power of 2 — the typical
    # training/eval case). The slice is a view; downstream einsum is
    # layout-agnostic, so no `.contiguous()` is needed.
    h = h_pad[:, :L] if L_pad != L else h_pad
    return h, scan_mode


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class _XlaFusedSSM(torch.autograd.Function):
    """Custom autograd.Function with analytic reverse-scan backward.

    Replaces the default autograd-through-the-scan backward, which would
    save every intermediate `(g, v)` pair from the parallel scan tree
    (~30–50 GB at 5M training shapes). The analytic backward saves only
    `(delta, A, B_proj, x, C_proj, D_param, h)` — about 14× smaller — and
    reconstructs `gates` plus runs ONE reverse scan to get `dh_acc`.

    Math (per (b, d, n) channel):

        Forward:   h[t] = g[t] · h[t-1] + tokens[t],    h[-1] = 0
                   y[t] = sum_n C[t,n] · h[t,n] + D[d] · x[t,d]

        Backward:  let r[t] = dy[t] · C[t]     (broadcast over d)
                   dh_acc[t] = C[t] · dy[t] + g[t+1] · dh_acc[t+1]
                   dh_acc[L-1] = C[L-1] · dy[L-1]

    The reverse recurrence is itself an associative scan with the same
    shape as forward. Reverse the time axis, shift gates left by 1
    (appending 0 at the end), and run the same `_run_scan` machinery.

    Per-input gradients then fall out from the chain rule:
        d_gate    = dh_acc · h_prev         (where h_prev = h shifted right)
        d_tokens  = dh_acc
        d_arg     = d_gate · gates · clamp_mask
        d_delta   = sum_n (d_arg · A + d_tokens · B · x)
        d_A       = sum_{B,L} d_arg · delta
        d_B_proj  = sum_d d_tokens · delta · x
        d_x       = D · dy + sum_n d_tokens · delta · B
        d_C_proj  = sum_d dy · h
        d_D_param = sum_{B,L} dy · x

    Reference for the math: Sasha Rush's Annotated Mamba (already cited
    in `mamba/fused_scan.py`) and the Triton `_fused_scan_bwd_kernel` in
    that file.
    """

    @staticmethod
    def forward(ctx, delta, A, B_proj, x, C_proj, D_param,
                scan_mode, chunk_size):
        out_dtype = x.dtype
        input_dtypes = (delta.dtype, A.dtype, B_proj.dtype,
                        x.dtype, C_proj.dtype, D_param.dtype)

        # Phase 5: do NOT force `.to(_COMPUTE_DTYPE)` on the six inputs.
        # Under `bf16: true` autocast the call site already passes A and
        # D_param as fp32 (`.float()` in mamba_llm_tpu.py:108-112) and
        # delta/B_proj/x/C_proj as bf16. Forcing fp32 on the bf16 inputs
        # was ~2.4 GiB/step of redundant HBM traffic at 5M training
        # shapes for no precision win — `arg = delta·A` already promotes
        # to fp32 via PyTorch's bf16×fp32 → fp32 rule, and the carry
        # inside `_chunked_scan` is held in fp32 explicitly.
        gates, tokens = _discretize(delta, A, B_proj, x)
        h, resolved_scan = _run_scan(gates, tokens, scan_mode, chunk_size)

        y = _einsum_promoted('bldn,bln->bld', h, C_proj) + D_param * x

        # Save inputs at their natural dtypes (saves ~140 MiB/scan-call
        # under bf16 autocast vs the previous always-fp32 storage); h
        # stays fp32 because the scan carry is fp32 and there's no
        # post-scan cast.
        ctx.save_for_backward(delta, A, B_proj, x, C_proj, D_param, h)
        ctx.scan_mode    = resolved_scan
        ctx.chunk_size   = chunk_size
        ctx.input_dtypes = input_dtypes
        ctx.out_dtype    = out_dtype
        return y.to(out_dtype)

    @staticmethod
    def backward(ctx, dy):
        (delta, A, B_proj, x, C_proj, D_param, h
         ) = ctx.saved_tensors
        scan_mode = ctx.scan_mode
        chunk_size = ctx.chunk_size
        input_dtypes = ctx.input_dtypes

        # Promote dy once. dy is small (B·L·D, ~33 MiB at 5M shapes) so
        # the upcast is cheap, and several of the gradient einsums (e.g.
        # d_D_param) would otherwise compute entirely in bf16 with no
        # fp32 operand to trigger promotion.
        dy_f = dy.to(_COMPUTE_DTYPE)

        # 1. Recompute gates from (delta, A). PyTorch promotion handles
        #    dtypes: `delta_bf16 × A_fp32 → fp32`, so `gates` is fp32
        #    even when delta arrived as bf16. Track the clamp mask so
        #    gradients vanish where exp's argument was clipped.
        arg = delta.unsqueeze(-1) * A                             # (B,L,D,N) fp32
        # Boundary-inclusive interior mask, matching `torch.clamp`'s own
        # autograd convention (gradient flows AT the boundary). In
        # production `delta > 0` (softplus) and `A < 0` (HiPPO init), so
        # `arg < 0` always and the upper boundary `arg == 0` is never
        # reached; making the bound inclusive is harmless here but
        # protects against a future refactor that allows `arg == 0`.
        clamp_mask = ((arg >= _CLAMP_LO) & (arg <= _CLAMP_HI)).to(arg.dtype)
        arg_clamped = torch.clamp(arg, min=_CLAMP_LO, max=_CLAMP_HI)
        gates = torch.exp(arg_clamped)                            # (B,L,D,N) fp32

        # 2. h_prev = h shifted right by 1 along time, leading slot zero.
        h_prev = torch.cat(
            [torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1,
        )                                                          # (B,L,D,N)

        # 3. r[t,d,n] = dy[t,d] · C[t,n]  (broadcast C over D dim).
        r = dy_f.unsqueeze(-1) * C_proj.unsqueeze(2)              # (B,L,D,N)

        # 4. Reverse-scan: shift gates left by 1, append a zero at the
        #    new last position, then flip along time. This puts the
        #    backward recurrence into the same form as the forward one,
        #    so we can call `_run_scan` unchanged.
        shifted_gates = torch.cat([
            gates[:, 1:, :, :],
            torch.zeros_like(gates[:, :1, :, :]),
        ], dim=1)
        rev_gates  = shifted_gates.flip(dims=[1])
        rev_tokens = r.flip(dims=[1])
        s, _ = _run_scan(rev_gates, rev_tokens, scan_mode, chunk_size)
        dh_acc = s.flip(dims=[1])                                 # (B,L,D,N) fp32

        # 5. Per-input gradients via the chain rule. dh_acc, gates and
        #    h_prev are fp32, so all derived terms below promote to fp32
        #    automatically.
        d_gate   = dh_acc * h_prev
        d_tokens = dh_acc
        d_arg    = d_gate * gates * clamp_mask                    # (B,L,D,N) fp32

        # delta gets contributions from both the gate and the token paths.
        d_delta_from_gate   = _einsum_promoted('bldn,dn->bld', d_arg, A)
        d_delta_from_tokens = (_einsum_promoted('bldn,bln->bld', d_tokens, B_proj)
                               * x)
        d_delta = d_delta_from_gate + d_delta_from_tokens

        d_A = _einsum_promoted('bldn,bld->dn', d_arg, delta)

        d_B_proj = _einsum_promoted('bldn,bld->bln', d_tokens, delta * x)

        # x: from token path AND from D feedthrough in the y projection.
        d_x_from_tokens = (_einsum_promoted('bldn,bln->bld', d_tokens, B_proj)
                           * delta)
        d_x_from_D      = D_param * dy_f
        d_x             = d_x_from_tokens + d_x_from_D

        d_C_proj  = _einsum_promoted('bld,bldn->bln', dy_f, h)
        d_D_param = _einsum_promoted('bld,bld->d', dy_f, x)

        # Cast each gradient back to its input's dtype.
        return (
            d_delta.to(input_dtypes[0]),
            d_A.to(input_dtypes[1]),
            d_B_proj.to(input_dtypes[2]),
            d_x.to(input_dtypes[3]),
            d_C_proj.to(input_dtypes[4]),
            d_D_param.to(input_dtypes[5]),
            None,         # scan_mode (non-tensor)
            None,         # chunk_size (non-tensor)
        )


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
        Backward is handled by an analytic reverse-scan custom autograd
        Function — see `_XlaFusedSSM` for the math. Saved-for-backward
        memory is ~14× smaller than the autograd-through-the-scan path
        this used to use.
    """
    return _XlaFusedSSM.apply(
        delta, A, B_proj, x, C_proj, D_param, scan, chunk_size,
    )
