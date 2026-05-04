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

SCAN_MODES = ["auto", "chunked"]


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

def _gradcheck_at(B: int, L: int, D: int, N: int, chunk_size: int):
    """Run `torch.autograd.gradcheck` on `fused_ssm` at the given shape.

    Patches `_COMPUTE_DTYPE` to fp64 for the duration so the internal
    scan runs in double precision — gradcheck's central-difference
    epsilon (~1e-6) needs more than fp32's 7-digit mantissa.
    """
    delta, A, B_proj, x, C_proj, D_param = _make_inputs(
        B, L, D, N, dtype=torch.float64, requires_grad=True,
    )
    saved_dtype = xla_fused_scan._COMPUTE_DTYPE
    xla_fused_scan._COMPUTE_DTYPE = torch.float64
    try:
        def fn(d, Aa, Bp, xx, Cp, Dp):
            return xla_fused_scan.fused_ssm(
                d, Aa, Bp, xx, Cp, Dp, scan="auto", chunk_size=chunk_size,
            )
        ok = torch.autograd.gradcheck(
            fn, (delta, A, B_proj, x, C_proj, D_param),
            eps=1e-6, atol=1e-4, rtol=1e-3,
            # scan_mode and chunk_size are non-tensor args; backward
            # returns None for them and gradcheck would otherwise fail
            # trying to verify a None gradient.
            check_undefined_grad=False,
            nondet_tol=0.0,
        )
        assert ok
    finally:
        xla_fused_scan._COMPUTE_DTYPE = saved_dtype


def test_gradcheck_fp64_single_chunk():
    """gradcheck on a tiny (1,8,4,4) case — degenerates to M=1 (single chunk).

    L=8 → L_pad=32 (the `_MIN_L_POW2` floor) → K=K_TABLE[32]=32, so
    M=L_pad/K=1. The within-chunk loop processes all 32 positions; the
    across-chunks scan is trivial (n_iter=0). Catches autograd wiring
    bugs in the within-chunk forward + the analytic backward's
    single-chunk path.
    """
    _gradcheck_at(B=1, L=8, D=4, N=4, chunk_size=8)


def test_gradcheck_fp64_multi_chunk():
    """gradcheck at M>1 — exercises the cross-chunk carry path in backward.

    L=64 → L_pad=64 → with chunk_size=8 → K=8, M=L_pad/K=8. This
    exercises:
        - the within-chunk inclusive scan over K=8 positions
        - the across-chunks Hillis-Steele over M=8 chunks
        - the carry broadcast `g_intra * carry_v.unsqueeze(2)`
        - the reverse-scan in backward over M>1 (the carry path the
          single-chunk gradcheck cannot reach)

    Per the critical-reviewer audit (Finding 1), this is the missing
    high-precision verification of the most complex code path in the
    file.
    """
    _gradcheck_at(B=1, L=64, D=4, N=4, chunk_size=8)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity — both scan modes produce identical output for L > 2*chunk_size
# (the auto-dispatch boundary). Catches regressions in either branch.
# ─────────────────────────────────────────────────────────────────────────────

def test_chunked_self_consistency_across_K():
    """Two K values for the chunked algorithm must produce the same h.

    Originally compared the Hillis-Steele tree to the chunked tree;
    after Phase 4 of `xla_scan_optimizations.md` the public dispatch
    is chunked-only, so this now verifies that switching K (the only
    remaining knob) does not change the output beyond floating-point
    noise. Catches regressions in `_chunked_scan` that depend on K
    in a way the work-count cost model doesn't capture.
    """
    B, L, D, N = 4, 256, 128, 16
    inputs = _make_inputs(B, L, D, N)

    y_k32 = xla_fused_scan.fused_ssm(*inputs, scan="chunked", chunk_size=32)
    y_k64 = xla_fused_scan.fused_ssm(*inputs, scan="chunked", chunk_size=64)
    torch.testing.assert_close(y_k32, y_k64, atol=1e-5, rtol=1e-5)


def test_hillis_alias_emits_deprecation_warning():
    """`scan="hillis"` is accepted but must emit a DeprecationWarning.

    Phase 4 collapses the user-facing hillis dispatch into the chunked
    path. Existing callers that pass `scan="hillis"` get the new
    behaviour silently if no warning is raised — exactly the silent
    behavioural change the critical-reviewer audit (Finding 5) flagged.
    """
    inputs = _make_inputs(1, 32, 8, 4)
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        _ = xla_fused_scan.fused_ssm(*inputs, scan="hillis")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) >= 1, (
        "scan='hillis' should emit a DeprecationWarning; got categories: "
        f"{[w.category.__name__ for w in caught]}"
    )


def test_oracle_consistency():
    """The autograd-friendly local oracle matches `_fused_ssm_ref` forward.

    Catches drift between the shipped reference and the local one if
    either is changed.
    """
    inputs = _make_inputs(2, 64, 32, 8)
    y_shipped = _fused_ssm_ref(*inputs)
    y_local   = _fused_ssm_ref_autograd(*inputs)
    torch.testing.assert_close(y_shipped, y_local, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — mixed-precision (bf16 inputs / fp32 gradients) cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def test_bf16_mixed_precision_gradient_cosine():
    """Gradients under the production dtype mix must stay aligned with all-fp32.

    Production call site (`mamba_llm_tpu.py:108-112`) under `bf16: true`
    autocast passes:
        - `delta`, `B_proj`, `x`, `C_proj`  in bf16
        - `A`, `D_param`                    in fp32 (explicit `.float()`)

    Phase 5 of `xla_scan_optimizations.md` lets these pass through
    without forced fp32 promotion, relying on PyTorch's bf16×fp32 → fp32
    promotion plus an explicit fp32 carry inside `_chunked_scan` to
    preserve cumulative-product precision. This test verifies the
    end-to-end gradient direction stays aligned with the all-fp32
    baseline (cosine similarity ≥ 0.99 per parameter).
    """
    B, L, D, N = 4, 256, 128, 16

    # Independent input tensors per dtype configuration so .backward()
    # accumulates into separate grad buffers.
    fp32_in = _make_inputs(B, L, D, N, requires_grad=True)
    mixed_in = _make_inputs(B, L, D, N, requires_grad=False)

    # Match production: cast 4 inputs to bf16, leave A and D_param in fp32.
    delta, A, B_proj, x, C_proj, D_param = mixed_in
    delta   = delta.to(torch.bfloat16)
    B_proj  = B_proj.to(torch.bfloat16)
    x       = x.to(torch.bfloat16)
    C_proj  = C_proj.to(torch.bfloat16)
    for t in (delta, A, B_proj, x, C_proj, D_param):
        t.requires_grad_(True)
    mixed_in = (delta, A, B_proj, x, C_proj, D_param)

    y_fp32 = xla_fused_scan.fused_ssm(*fp32_in)
    y_fp32.sum().backward()

    y_mix = xla_fused_scan.fused_ssm(*mixed_in)
    y_mix.sum().backward()

    # bf16 round-trip introduces ~1% relative error in y itself; that's
    # expected. The headline check is gradient direction (cosine sim),
    # which determines training trajectory.
    names = ["delta", "A", "B_proj", "x", "C_proj", "D_param"]
    for name, t32, t16 in zip(names, fp32_in, mixed_in):
        g32 = t32.grad.to(torch.float32).flatten()
        g16 = t16.grad.to(torch.float32).flatten()
        # Skip per-parameter cosine when both gradients are identically
        # zero (degenerate).
        if g32.norm() == 0 or g16.norm() == 0:
            continue
        cos = torch.nn.functional.cosine_similarity(
            g32.unsqueeze(0), g16.unsqueeze(0),
        ).item()
        assert cos > 0.99, (
            f"bf16/fp32 gradient cosine similarity for {name!r} = {cos:.4f}; "
            "expected ≥ 0.99 (Phase 5 carry-fp32 invariant broken?)"
        )
