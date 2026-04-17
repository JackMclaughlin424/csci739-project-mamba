# Sources:
# Based heavily on Sasha Rush's Mamba: The Hard Way
# https://srush.github.io/annotated-mamba/hard.html
# https://github.com/srush/annotated-mamba

# Generative AI was used in the implementation of this file
# to help correct the original implementation (I don't know Triton very well), 
# optimize performance, and document the code.

"""
Fused Triton kernels for Mamba SSM: discretize + parallel scan.

Forward kernel  — grid (B, D*N): fused discretize + associative scan → h
Backward kernel — grid (B, D) with N loop: reverse scan + gradient accumulation
                  All (B,D,N,L) intermediates stay in registers; never materialised.

Eliminates ~4 GB of backward intermediate tensors vs the unfused approach.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════════════
# Triton kernels
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _combine_fn(g_a, v_a, g_b, v_b):
        """First-order affine composition: h = g·h_prev + v"""
        return g_a * g_b, v_a * g_b + v_b

    # ── Forward kernel ────────────────────────────────────────────────────
    # Grid: (B, D*N).  All inputs channel-first (B, *, L) for coalesced access.

    @triton.jit
    def _fused_scan_fwd_kernel(
        delta_ptr,      # (B, D, L)
        A_ptr,          # (D, N)
        B_ptr,          # (B, N, L)
        x_ptr,          # (B, D, L)
        h_ptr,          # (B, D*N, L) output
        D_dim, N_dim, L,
        BLOCK_L: tl.constexpr,
    ):
        b  = tl.program_id(0)
        dn = tl.program_id(1)
        d  = dn // N_dim
        n  = dn % N_dim

        ls   = tl.arange(0, BLOCK_L)
        mask = ls < L

        delta = tl.load(delta_ptr + (b * D_dim + d) * L + ls, mask=mask, other=0.0)
        x_val = tl.load(x_ptr    + (b * D_dim + d) * L + ls, mask=mask, other=0.0)
        a_val = tl.load(A_ptr + d * N_dim + n)
        b_val = tl.load(B_ptr + (b * N_dim + n) * L + ls, mask=mask, other=0.0)

        gate  = tl.where(mask, tl.exp(delta * a_val), 0.0)
        token = tl.where(mask, delta * b_val * x_val, 0.0)
        _, h_val = tl.associative_scan((gate, token), 0, _combine_fn)

        tl.store(h_ptr + (b * D_dim * N_dim + dn) * L + ls, h_val, mask=mask)

    # ── Output projection kernel ─────────────────────────────────────────
    # Grid: (B, D).  Replaces torch.einsum('bdnl,bnl->bdl', h, C) + D*x
    # Reads C in original (B, L, N) layout (stride-N) to skip the
    # .transpose().contiguous() copy.

    @triton.jit
    def _output_proj_kernel(
        h_ptr,          # (B, D*N, L) contiguous — from scan kernel
        C_ptr,          # (B, L, N)  contiguous — original layout, stride-N reads
        x_ptr,          # (B, D, L)  contiguous — channel-first
        D_ptr,          # (D,)
        y_ptr,          # (B, D, L)  output — channel-first
        D_dim, N_dim, L,
        BLOCK_L: tl.constexpr,
        N: tl.constexpr,
    ):
        b = tl.program_id(0)
        d = tl.program_id(1)

        ls   = tl.arange(0, BLOCK_L)
        mask = ls < L

        # D feedthrough: y = D[d] · x[b,d,:]
        x_val = tl.load(x_ptr + (b * D_dim + d) * L + ls, mask=mask, other=0.0)
        d_val = tl.load(D_ptr + d)
        y_acc = d_val * x_val

        for n in tl.static_range(N):
            # h[b, d*N+n, :]  — stride-1 (coalesced)
            h_val = tl.load(
                h_ptr + (b * D_dim * N_dim + d * N_dim + n) * L + ls,
                mask=mask, other=0.0,
            )
            # C[b, :, n]  — stride-N from (B, L, N) layout
            c_val = tl.load(
                C_ptr + b * L * N_dim + ls * N_dim + n,
                mask=mask, other=0.0,
            )
            y_acc += c_val * h_val

        tl.store(y_ptr + (b * D_dim + d) * L + ls, y_acc, mask=mask)

    # ── Backward kernel ───────────────────────────────────────────────────
    # Grid: (B, D) with tl.static_range(N) inner loop.
    # Reverse scan + gradient accumulation per (b,d,n) — all intermediates
    # (gate, dh, dh_acc, h_prev, d_gate, d_token) live in registers only.
    #
    # Outputs written directly:  ddelta (B,D,L), dx (B,D,L)
    # Outputs via atomic_add:    dB (B,N,L), dC (B,N,L), dA (D,N), dD (D,)

    @triton.jit
    def _fused_scan_bwd_kernel(
        # Saved from forward (channel-first, contiguous)
        delta_ptr,      # (B, D, L)
        A_ptr,          # (D, N)
        B_ptr,          # (B, N, L)
        x_ptr,          # (B, D, L)
        C_ptr,          # (B, N, L)
        D_ptr,          # (D,)
        h_ptr,          # (B, D*N, L)
        # Gradient input
        dy_ptr,         # (B, D, L)
        # Gradient outputs — direct writes
        ddelta_ptr,     # (B, D, L)
        dx_ptr,         # (B, D, L)
        # Gradient outputs — atomic adds
        dB_ptr,         # (B, N, L)  — zeroed before launch
        dC_ptr,         # (B, N, L)  — zeroed before launch
        dA_ptr,         # (D, N)     — zeroed before launch
        dD_ptr,         # (D,)       — zeroed before launch
        # Dimensions
        D_dim, N_dim, L,
        BLOCK_L: tl.constexpr,
        N: tl.constexpr,
    ):
        b = tl.program_id(0)
        d = tl.program_id(1)

        ls   = tl.arange(0, BLOCK_L)
        mask = ls < L

        # ── Coalesced loads shared across all N iterations ────────────────
        dl_off = (b * D_dim + d) * L
        delta  = tl.load(delta_ptr + dl_off + ls, mask=mask, other=0.0)
        x_val  = tl.load(x_ptr    + dl_off + ls, mask=mask, other=0.0)
        dy_val = tl.load(dy_ptr   + dl_off + ls, mask=mask, other=0.0)
        d_feed = tl.load(D_ptr + d)

        # Accumulators for ddelta and dx (summed across N)
        ddelta_acc = tl.zeros([BLOCK_L], dtype=tl.float32)
        dx_acc     = d_feed * dy_val          # D·dy contribution

        # Shifted delta for gate_next: delta[l+1], 0 at end
        mask_next  = mask & (ls < L - 1)
        delta_next = tl.load(delta_ptr + dl_off + ls + 1,
                             mask=mask_next, other=0.0)

        for n in tl.static_range(N):
            a_val = tl.load(A_ptr + d * N_dim + n)

            bn_off = (b * N_dim + n) * L
            b_val  = tl.load(B_ptr + bn_off + ls, mask=mask, other=0.0)
            c_val  = tl.load(C_ptr + bn_off + ls, mask=mask, other=0.0)

            # h[b, d*N+n, :]
            h_off  = (b * D_dim * N_dim + d * N_dim + n) * L
            h_val  = tl.load(h_ptr + h_off + ls, mask=mask, other=0.0)

            # ── Recompute gate on-the-fly ─────────────────────────────────
            gate = tl.where(mask, tl.exp(delta * a_val), 0.0)

            # ── dh from output projection: dh = dy · C ───────────────────
            dh = dy_val * c_val

            # ── Shifted gate for reverse scan: gate_next[l] = gate[l+1] ──
            gate_next = tl.where(mask_next, tl.exp(delta_next * a_val), 0.0)

            # ── Reverse associative scan ──────────────────────────────────
            # dh_acc[t] = gate_next[t]·dh_acc[t+1] + dh[t]
            _, dh_acc = tl.associative_scan(
                (gate_next, dh), 0, _combine_fn, reverse=True
            )

            # ── h_prev[l] = h[l-1],  h_prev[0] = 0 ──────────────────────
            mask_prev = mask & (ls > 0)
            h_prev = tl.load(h_ptr + h_off + ls - 1,
                             mask=mask_prev, other=0.0)

            # ── Per-(b,d,n) gradients ─────────────────────────────────────
            d_gate  = dh_acc * h_prev
            d_token = dh_acc

            # ── Accumulate ddelta, dx (local — no atomics) ───────────────
            # ddelta += d_gate·gate·A + d_token·B·x
            ddelta_acc += d_gate * gate * a_val + d_token * b_val * x_val
            # dx += d_token·delta·B
            dx_acc += d_token * delta * b_val

            # ── Atomic adds for cross-D reductions ───────────────────────
            # dB[b,n,:] += d_token · delta · x    (D contenders per address)
            tl.atomic_add(dB_ptr + bn_off + ls,
                          d_token * delta * x_val, mask=mask)
            # dC[b,n,:] += dy · h                 (D contenders per address)
            tl.atomic_add(dC_ptr + bn_off + ls,
                          dy_val * h_val, mask=mask)
            # dA[d,n] += sum_l(d_gate·gate·delta) (B contenders)
            dA_contrib = tl.sum(d_gate * gate * delta, axis=0)
            tl.atomic_add(dA_ptr + d * N_dim + n, dA_contrib)

        # ── Store ddelta[b,d,:] and dx[b,d,:] (direct write) ────────────
        tl.store(ddelta_ptr + dl_off + ls, ddelta_acc, mask=mask)
        tl.store(dx_ptr     + dl_off + ls, dx_acc,     mask=mask)

        # dD[d] += sum_l(dy · x)  (B contenders)
        tl.atomic_add(dD_ptr + d, tl.sum(dy_val * x_val, axis=0))


# ═══════════════════════════════════════════════════════════════════════════
# Autograd wrapper
# ═══════════════════════════════════════════════════════════════════════════

def _next_pow2(n):
    """Next power of 2 >= max(32, n)."""
    return max(32, 1 << (n - 1).bit_length())


if HAS_TRITON:

    class _FusedSSM(torch.autograd.Function):

        @staticmethod
        def forward(ctx, delta, A, B_proj, x, C_proj, D_param):
            B_dim, L, D_dim = delta.shape
            N = A.shape[1]
            DN = D_dim * N
            BLOCK_L = _next_pow2(L)

            # Transpose to channel-first for coalesced kernel access
            delta_t = delta.transpose(1, 2).contiguous()    # (B, D, L)
            x_t     = x.transpose(1, 2).contiguous()        # (B, D, L)
            B_t     = B_proj.transpose(1, 2).contiguous()   # (B, N, L)
            C_t     = C_proj.transpose(1, 2).contiguous()   # (B, N, L) — for backward

            # Fused discretize + scan → h
            h = torch.empty(B_dim, DN, L, device=x.device, dtype=torch.float32)

            _fused_scan_fwd_kernel[(B_dim, DN)](
                delta_t, A, B_t, x_t,
                h,
                D_dim, N, L,
                BLOCK_L=BLOCK_L,
            )

            # Fused output projection: y = sum_n(C·h) + D·x
            y_t = torch.empty(B_dim, D_dim, L, device=x.device, dtype=torch.float32)

            _output_proj_kernel[(B_dim, D_dim)](
                h, C_proj, x_t, D_param,
                y_t,
                D_dim, N, L,
                BLOCK_L=BLOCK_L, N=N,
            )

            y = y_t.transpose(1, 2)                  # (B, L, D) — free view

            # Save channel-first tensors for backward
            ctx.save_for_backward(delta_t, A, B_t, x_t, C_t, D_param, h)
            ctx.shapes = (B_dim, L, D_dim, N)
            ctx.BLOCK_L = BLOCK_L
            return y

        @staticmethod
        def backward(ctx, dy):
            delta_t, A, B_t, x_t, C_t, D_param, h = ctx.saved_tensors
            B_dim, L, D_dim, N = ctx.shapes
            BLOCK_L = ctx.BLOCK_L

            dy_t = dy.transpose(1, 2).contiguous()    # (B, D, L)

            # Allocate gradient outputs
            ddelta_t = torch.empty_like(delta_t)      # (B, D, L)
            dx_t     = torch.empty_like(x_t)          # (B, D, L)
            dB_t     = torch.zeros_like(B_t)          # (B, N, L) — zeroed for atomics
            dC_t     = torch.zeros_like(C_t)          # (B, N, L) — zeroed for atomics
            dA       = torch.zeros_like(A)            # (D, N)    — zeroed for atomics
            dD       = torch.zeros(D_dim, device=A.device, dtype=torch.float32)

            _fused_scan_bwd_kernel[(B_dim, D_dim)](
                delta_t, A, B_t, x_t, C_t, D_param, h,
                dy_t,
                ddelta_t, dx_t,
                dB_t, dC_t, dA, dD,
                D_dim, N, L,
                BLOCK_L=BLOCK_L, N=N,
            )

            # Transpose gradients back to (B, L, *) layout
            return (
                ddelta_t.transpose(1, 2),   # (B, L, D)
                dA,                          # (D, N)
                dB_t.transpose(1, 2),        # (B, L, N)
                dx_t.transpose(1, 2),        # (B, L, D)
                dC_t.transpose(1, 2),        # (B, L, N)
                dD,                          # (D,)
            )


# ═══════════════════════════════════════════════════════════════════════════
# Reference (pure-PyTorch) fallback
# ═══════════════════════════════════════════════════════════════════════════

def _fused_ssm_ref(delta, A, B_proj, x, C_proj, D_param):
    """Sequential scan — no Triton, CPU-safe, always correct."""
    batch, L, d_model = x.shape
    d_state = A.shape[1]

    delta_t = delta.transpose(1, 2)
    x_t     = x.transpose(1, 2)
    B_t     = B_proj.transpose(1, 2)
    C_t     = C_proj.transpose(1, 2)

    gates  = torch.exp(delta_t.unsqueeze(2) * A.unsqueeze(-1))
    tokens = delta_t.unsqueeze(2) * B_t.unsqueeze(1) * x_t.unsqueeze(2)

    h = torch.zeros(batch, d_model, d_state, L, device=x.device, dtype=x.dtype)
    for t in range(L):
        if t == 0:
            h[:, :, :, t] = tokens[:, :, :, t]
        else:
            h[:, :, :, t] = gates[:, :, :, t] * h[:, :, :, t - 1] + tokens[:, :, :, t]

    y = torch.einsum('bdnl,bnl->bdl', h, C_t) + D_param.unsqueeze(-1) * x_t
    return y.transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def fused_ssm(delta, A, B_proj, x, C_proj, D_param):
    """Fused SSM: discretize + parallel scan + output projection.

    Args:
        delta:  (B, L, D) timestep sizes (after softplus)
        A:      (D, N)    state decay matrix (negative)
        B_proj: (B, L, N) input projection
        x:      (B, L, D) input signal
        C_proj: (B, L, N) output projection
        D_param:(D,)      feedthrough

    Returns:
        y: (B, L, D) SSM output
    """
    if HAS_TRITON and delta.is_cuda:
        return _FusedSSM.apply(
            delta.contiguous().float(),
            A.contiguous().float(),
            B_proj.contiguous().float(),
            x.contiguous().float(),
            C_proj.contiguous().float(),
            D_param.contiguous().float(),
        )
    return _fused_ssm_ref(delta, A, B_proj, x, C_proj, D_param)
