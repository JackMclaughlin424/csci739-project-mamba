"""Parity tests for `mamba.xla_fused_scan.fused_ssm`.

Oracle: `mamba.fused_scan._fused_ssm_ref`, the pure-PyTorch sequential
scan that powers the existing CUDA/Triton kernel's reference path.

These tests run on CPU and complete in a few seconds. They are the
parity gate every later phase of `xla_scan_optimizations.md` must
continue to pass:

    pytest tests/test_xla_fused_scan.py -v

Tolerances are intentionally loose enough to absorb the legitimate
differences between the oracle (no clamp on `δ·A`) and `xla_fused_scan`
(`clamp(arg, -20, 0)`), but tight enough to fail on any algorithmic
drift. To keep the clamp from firing, all test inputs bound `|A|` away
from large magnitudes.
"""

import math

import pytest
import torch

from mamba import xla_fused_scan
from mamba.fused_scan import _fused_ssm_ref


def _fused_ssm_ref_autograd(delta, A, B_proj, x, C_proj, D_param):
    """Autograd-friendly mirror of `mamba.fused_scan._fused_ssm_ref`.

    The shipped oracle writes `h[..., t] = ...` in a Python loop, which is
    an in-place op that newer PyTorch refuses to backward through. This
    builds the same `h` via `torch.stack` over a list — semantically
    identical for forward, but differentiable.
    """
    batch, L, d_model = x.shape
    d_state = A.shape[1]

    delta_t = delta.transpose(1, 2)
    x_t     = x.transpose(1, 2)
    B_t     = B_proj.transpose(1, 2)
    C_t     = C_proj.transpose(1, 2)

    gates  = torch.exp(delta_t.unsqueeze(2) * A.unsqueeze(-1))         # (B,D,N,L)
    tokens = delta_t.unsqueeze(2) * B_t.unsqueeze(1) * x_t.unsqueeze(2)

    h_list = []
    h_prev = torch.zeros(batch, d_model, d_state,
                         device=x.device, dtype=x.dtype)
    for t in range(L):
        h_t = gates[:, :, :, t] * h_prev + tokens[:, :, :, t]
        h_list.append(h_t)
        h_prev = h_t
    h = torch.stack(h_list, dim=-1)                                    # (B,D,N,L)

    y = torch.einsum('bdnl,bnl->bdl', h, C_t) + D_param.unsqueeze(-1) * x_t
    return y.transpose(1, 2)


# Mirror `_build_debug_notebook.py:280-291` test-vector shapes plus a
# single full-d_model case. Kept small so CPU runs in under a second.
SHAPES = [
    # (B, L, D, N)
    (2,  64,  64, 16),
    (4, 128, 128, 16),
    (1, 256, 256, 16),
    (8, 512,  64, 16),
    (1, 512, 512, 16),
]

SCAN_MODES = ["hillis", "chunked"]


# ─────────────────────────────────────────────────────────────────────────────
# Input fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_inputs(B: int, L: int, D: int, N: int,
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0,
                 requires_grad: bool = False):
    """Random Mamba-shaped inputs with magnitudes bounded so `arg = δ·A`
    stays well above the `-20` clamp floor that `xla_fused_scan` applies
    but `_fused_ssm_ref` does not.

    With `δ ~ U(0,1)` and `|A| ∈ [0.1, 1.1]`, max |arg| ≤ 1.1 — clamp
    cannot fire, so any output drift is real (not a clamp artifact).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    delta   = torch.rand(B, L, D, generator=g, dtype=dtype)
    A       = -(0.1 + torch.rand(D, N, generator=g, dtype=dtype))   # negative
    B_proj  = torch.randn(B, L, N, generator=g, dtype=dtype) * 0.5
    x       = torch.randn(B, L, D, generator=g, dtype=dtype) * 0.5
    C_proj  = torch.randn(B, L, N, generator=g, dtype=dtype) * 0.5
    D_param = torch.randn(D, generator=g, dtype=dtype) * 0.5

    if requires_grad:
        for t in (delta, A, B_proj, x, C_proj, D_param):
            t.requires_grad_(True)

    return delta, A, B_proj, x, C_proj, D_param


# ─────────────────────────────────────────────────────────────────────────────
# Forward parity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BLDN={s}")
@pytest.mark.parametrize("scan", SCAN_MODES)
def test_forward_parity(shape, scan):
    """`fused_ssm(...)` ≈ `_fused_ssm_ref(...)` to atol/rtol=1e-4 in fp32."""
    B, L, D, N = shape
    delta, A, B_proj, x, C_proj, D_param = _make_inputs(B, L, D, N)

    y_ref = _fused_ssm_ref(delta, A, B_proj, x, C_proj, D_param)
    y_xla = xla_fused_scan.fused_ssm(
        delta, A, B_proj, x, C_proj, D_param,
        scan=scan, chunk_size=min(128, L),
    )

    assert y_xla.shape == y_ref.shape, (y_xla.shape, y_ref.shape)
    assert y_xla.dtype == y_ref.dtype
    torch.testing.assert_close(y_xla, y_ref, atol=1e-4, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Backward parity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BLDN={s}")
@pytest.mark.parametrize("scan", SCAN_MODES)
def test_backward_parity(shape, scan):
    """Per-input gradients agree to atol=1e-3, rtol=1e-3.

    Looser than forward because backward accumulates more rounding via
    PyTorch autograd through the parallel-scan tree.
    """
    B, L, D, N = shape

    # Build two independent copies of the inputs so .backward() doesn't
    # share grad buffers between the oracle and the candidate.
    ref_inputs = _make_inputs(B, L, D, N, requires_grad=True)
    xla_inputs = _make_inputs(B, L, D, N, requires_grad=True)

    y_ref = _fused_ssm_ref_autograd(*ref_inputs)
    y_ref.sum().backward()

    y_xla = xla_fused_scan.fused_ssm(
        *xla_inputs, scan=scan, chunk_size=min(128, L),
    )
    y_xla.sum().backward()

    names = ["delta", "A", "B_proj", "x", "C_proj", "D_param"]
    for name, ti_ref, ti_xla in zip(names, ref_inputs, xla_inputs):
        assert ti_ref.grad is not None, f"reference produced no grad for {name}"
        assert ti_xla.grad is not None, f"xla produced no grad for {name}"
        torch.testing.assert_close(
            ti_xla.grad, ti_ref.grad, atol=1e-3, rtol=1e-3,
            msg=lambda m, n=name: f"gradient mismatch for {n}: {m}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Autograd wiring (gradcheck) — fp64 small case
# ─────────────────────────────────────────────────────────────────────────────

def test_gradcheck_fp64_small():
    """`torch.autograd.gradcheck` on a tiny (1,8,4,4) fp64 case.

    Catches autograd wiring bugs (wrong number of returned grads, wrong
    shape, missing detach, etc.) that the parity tests above can miss
    when the wrong-but-self-consistent gradient happens to numerically
    match the oracle's wrong-but-self-consistent gradient.

    `_COMPUTE_DTYPE` is monkey-patched to fp64 for the duration so the
    internal scan runs in double precision — gradcheck's central-
    difference epsilon (~1e-6) needs more than fp32's 7-digit mantissa.
    """
    B, L, D, N = 1, 8, 4, 4
    delta, A, B_proj, x, C_proj, D_param = _make_inputs(
        B, L, D, N, dtype=torch.float64, requires_grad=True,
    )

    saved_dtype = xla_fused_scan._COMPUTE_DTYPE
    xla_fused_scan._COMPUTE_DTYPE = torch.float64
    try:
        # `torch.autograd.gradcheck` calls fused_ssm many times with
        # perturbed inputs; chunked scan with K=L (one chunk) is
        # numerically simplest and shares a code path with hillis at L=8.
        def fn(d, Aa, Bp, xx, Cp, Dp):
            return xla_fused_scan.fused_ssm(
                d, Aa, Bp, xx, Cp, Dp, scan="hillis", chunk_size=8,
            )

        ok = torch.autograd.gradcheck(
            fn, (delta, A, B_proj, x, C_proj, D_param),
            eps=1e-6, atol=1e-4, rtol=1e-3,
            check_undefined_grad=False,    # D_param is summed, that's expected
            nondet_tol=0.0,
        )
        assert ok
    finally:
        xla_fused_scan._COMPUTE_DTYPE = saved_dtype


# ─────────────────────────────────────────────────────────────────────────────
# Sanity — both scan modes produce identical output for L > 2*chunk_size
# (the auto-dispatch boundary). Catches regressions in either branch.
# ─────────────────────────────────────────────────────────────────────────────

def test_hillis_vs_chunked_self_consistency():
    """Both scan implementations must produce the same forward output."""
    B, L, D, N = 4, 256, 128, 16
    inputs = _make_inputs(B, L, D, N)

    y_hillis  = xla_fused_scan.fused_ssm(*inputs, scan="hillis",  chunk_size=64)
    y_chunked = xla_fused_scan.fused_ssm(*inputs, scan="chunked", chunk_size=64)
    torch.testing.assert_close(y_hillis, y_chunked, atol=1e-5, rtol=1e-5)


def test_oracle_consistency():
    """The autograd-friendly local oracle matches `_fused_ssm_ref` forward.

    Catches drift between the shipped reference and the local one if
    either is changed.
    """
    inputs = _make_inputs(2, 64, 32, 8)
    y_shipped = _fused_ssm_ref(*inputs)
    y_local   = _fused_ssm_ref_autograd(*inputs)
    torch.testing.assert_close(y_shipped, y_local, atol=1e-5, rtol=1e-5)
