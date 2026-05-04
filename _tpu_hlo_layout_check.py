"""One-shot HLO layout-copy check for `mamba/xla_fused_scan.py`.

Decision gate for Phase 3 of `xla_scan_optimizations.md` (eliminate the
three `(B,L,D,N)` permute/contiguous copies).

The audit predicts that `xla_fused_scan.py:225,226,251` materialize as three
distinct `copy` HLO ops, each moving ~2 GiB of HBM at 5M training shapes
(B=128, L=512, D=512, N=16; ~6 GiB total per scan call). XLA's
layout-assignment pass *can sometimes* elide these. If it already does,
Phase 3 is a no-op and should be skipped; if it doesn't, Phase 3 is the
biggest layout-traffic win available.

Usage on the TPU VM:

    XLA_FLAGS="--xla_dump_to=/tmp/hlo_layout_check --xla_dump_hlo_as_text" \\
    PJRT_DEVICE=TPU python _tpu_hlo_layout_check.py

    grep -c 'copy(' /tmp/hlo_layout_check/*.txt | sort -t: -k2 -nr | head
    grep -E 'copy|transpose' /tmp/hlo_layout_check/module_*.txt | wc -l

Decision rule:
    * Three or more `copy` ops on `f32[128,512,512,16]`-shaped tensors → Phase 3 is real; pursue.
    * Zero to one such copy → XLA already elides; Phase 3 priority drops sharply.

The script triggers exactly ONE forward pass through `fused_ssm` at
training shapes; that's all XLA needs to dump the HLO module.
"""

import os
import sys

import torch

# Make the repo root importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba.xla_fused_scan import fused_ssm

# 5M training-config shapes. Mirror config_5M.yaml exactly.
B = 128
L = 512
D_MODEL = 512
N = 16


def main() -> int:
    if "XLA_FLAGS" not in os.environ or "xla_dump_to" not in os.environ.get("XLA_FLAGS", ""):
        print("WARNING: XLA_FLAGS does not set --xla_dump_to. The HLO dump "
              "will not be written. Re-run with:", file=sys.stderr)
        print('  XLA_FLAGS="--xla_dump_to=/tmp/hlo_layout_check '
              '--xla_dump_hlo_as_text" PJRT_DEVICE=TPU '
              'python _tpu_hlo_layout_check.py', file=sys.stderr)

    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        import torch_xla as _txla
        device = _txla.device()
        sync = getattr(_txla, "sync", None) or xm.mark_step
    except ImportError:
        print("torch_xla not available; running on CPU (HLO dump will not "
              "include TPU layout decisions). This is only useful as a "
              "smoke test.", file=sys.stderr)
        device = torch.device("cpu")
        sync = lambda: None

    print(f"Running fused_ssm at shapes (B={B}, L={L}, D={D_MODEL}, N={N}) "
          f"on {device}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(0)
    delta   = torch.rand(B, L, D_MODEL, generator=g).to(device)
    A       = (-torch.rand(D_MODEL, N, generator=g).abs()).to(device)
    B_proj  = torch.randn(B, L, N, generator=g).to(device)
    x       = torch.randn(B, L, D_MODEL, generator=g).to(device)
    C_proj  = torch.randn(B, L, N, generator=g).to(device)
    D_param = torch.randn(D_MODEL, generator=g).to(device)

    y = fused_ssm(delta, A, B_proj, x, C_proj, D_param)
    sync()
    print(f"forward done; y.shape={tuple(y.shape)}, y.dtype={y.dtype}",
          flush=True)
    print("HLO modules should now be in the directory passed to "
          "--xla_dump_to. Inspect with:", flush=True)
    print("  grep -E 'copy|transpose' <dump_dir>/module_*.txt | wc -l",
          flush=True)
    print("  grep -E 'f32\\[128,512,512,16\\]' <dump_dir>/module_*.txt | "
          "head -20", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
