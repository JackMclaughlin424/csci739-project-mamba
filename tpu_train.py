"""
Mamba LM training script — SimpleStories-5M-equivalent for TPU v4 / v6e.

Trains a ~5M-parameter Mamba LM on the HuggingFace SimpleStories corpus,
param-matched to the SimpleStories-5M transformer baseline.

Architecture follows the canonical Mamba recipe (Gu & Dao 2023):
inner expansion E=2, more layers than the transformer to hit the same
param budget — the paper's consistent finding is that Mamba prefers depth
over width.

    | spec value     | maps to              | this script default |
    | d_model = 256  | residual stream      | d_input = 256       |
    | n_ctx = 512    | context length       | seq_len = 512       |
    | d_vocab = 4096 | vocabulary size      | vocab_size = 4096   |
    | n_layers = 6   | (transformer-spec)   | n_layer = 9 (Mamba) |
    | n_params ≈ 5M  | unique parameters    | d_model = 512 (E=2) |
    |                |                      | dt_rank = 32 (=d/16)|
    |                |                      |   → 5.14M unique    |

Pass --n_layer 6 --d_model 768 for the depth-matched (E=3) variant if you
want to isolate "depth or architecture wins?" in the comparison study.

Hardware: single-host TPU v4-8 / v6e-{1,4,8} (SPMD across all cores in a
SINGLE Python process — no xmp.spawn). Falls back to CUDA / CPU when
torch_xla is missing, so the same script runs in local dev and on TPU.

Single-device (CPU / single TPU core / dev):
    python tpu_train.py

Multi-device on TPU (single Python process, GSPMD partitioner shards the
global batch across all available cores):
    PJRT_DEVICE=TPU python tpu_train.py --multi_device

Why SPMD over xmp.spawn:
    * One Python interpreter (not N) — shared in-process XLA compile cache
    * No per-rank cold-compile cost (~770 s × N saved)
    * args.batch_size is per-device; the script multiplies by num_devices
      for the DataLoader so the global batch is correctly assembled, then
      mark_sharding(batch, mesh, ('data', None)) re-distributes per device
    * Required to scale beyond a single host (TPU pods are SPMD only)

XLA best practices applied (see `mamba/xla_tpu_reference.md`):
    * MpDeviceLoader (§2.4) — async host→device prefetch
    * torch_xla.amp.syncfree.AdamW (§9.2) — no internal .item() syncs
    * xm.optimizer_step (§9.1) — consolidates gradients + implicit mark_step
    * On-device NaN guard (§9.3) — no per-step .item() on isfinite checks
    * Diagnostics computed every step, transferred only at log_interval (§9.4)
    * int32 stored token ids (§8.2) — cast to long only at the embedding
    * GSPMD batch sharding — mark_sharding(batch, mesh, ('data', None))
      replicates model + shards data dim across all XLA devices
    * model.step graph pre-compiled at startup so end-of-epoch sample
      generation doesn't pay a cold-compile stall mid-training

Optional environment variables for TPU launches:
    PJRT_DEVICE=TPU
    XLA_PERSISTENT_CACHE_PATH=/path/to/cache       # cache compiled programs
    PT_XLA_DEBUG_LEVEL=2                           # surface compile/sync issues

Note: XLA_PERSISTENT_CACHE_PATH has a known load bug on torch_xla 2.9.0
(xla_tpu_reference §1.5) — programs write but don't reload.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── XLA imports with graceful CPU/CUDA fallback ─────────────────────────────
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.debug.metrics as met
    from torch_xla.amp.syncfree import AdamW as SyncfreeAdamW
    HAS_XLA = True

    # ── API compatibility shims ────────────────────────────────────────────
    # torch_xla 2.5+ removed `xm.xrt_world_size`, `xm.is_master_ordinal`, and
    # deprecated `xm.xla_device` in favour of the `torch_xla.runtime` namespace
    # and the top-level `torch_xla.device()`. Older releases (≤2.4) only have
    # the `xm.*` API. Resolve once here so the rest of the file is version-clean.
    try:
        import torch_xla.runtime as xr
        import torch_xla as _txla
        _xla_world_size    = xr.world_size
        _xla_process_index = xr.process_index
        _xla_is_master     = lambda: xr.process_index() == 0
        _xla_device        = _txla.device
    except (ImportError, AttributeError):
        _xla_world_size    = xm.xrt_world_size
        _xla_process_index = xm.get_ordinal
        _xla_is_master     = xm.is_master_ordinal
        _xla_device        = xm.xla_device
    # `xm.mark_step()` was deprecated in torch_xla 2.5+ in favour of
    # `torch_xla.sync()`. Resolve once and use `_xla_sync()` site-wide.
    _xla_sync = getattr(_txla, "sync", None) or xm.mark_step
except ImportError:
    HAS_XLA = False
    SyncfreeAdamW = torch.optim.AdamW
    xm = met = pl = None  # type: ignore
    _xla_world_size = _xla_process_index = _xla_is_master = _xla_device = None  # type: ignore
    _xla_sync = lambda: None  # type: ignore

# SPMD detection — torch_xla 2.0+ ships the GSPMD partitioner under
# `torch_xla.distributed.spmd`. Required for the single-process / mark-
# sharding execution model that replaced `xmp.spawn` in this script.
HAS_SPMD = False
xs = None  # type: ignore
if HAS_XLA:
    try:
        import torch_xla.distributed.spmd as _xs
        if hasattr(xr, "use_spmd") and hasattr(_xs, "Mesh") and hasattr(_xs, "mark_sharding"):
            xs = _xs
            HAS_SPMD = True
    except ImportError:
        HAS_SPMD = False

# ─── Optional dependencies for HF data loading ───────────────────────────────
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore
        return it

# ─── Optional wandb (graceful fallback) ──────────────────────────────────────
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # type: ignore

from mamba.mamba_llm_tpu import MambaLMHeadModel, MambaLMConfig


# ─────────────────────────────────────────────────────────────────────────────
# wandb artifact helpers — cache the tokenized corpus across runs
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(s: str) -> str:
    """wandb artifact names allow [A-Za-z0-9._-]; replace everything else."""
    out = []
    for c in s:
        out.append(c if (c.isalnum() or c in "._-") else "-")
    return "".join(out).strip("-").lower() or "x"


def _tokens_artifact_name(args) -> str:
    """Auto-derived artifact name; stable for a given (dataset, tokenizer, case, seq_len, n)."""
    if args.tokens_artifact:
        return args.tokens_artifact
    ds   = _slugify(args.dataset_name)
    tok  = _slugify(args.tokenizer_name)
    n    = "full" if not args.max_stories else f"n{args.max_stories}"
    case = "lc" if args.lowercase else "tc"          # lowercased / true-case
    sl   = f"s{args.seq_len}"
    # v3 = packed (N, seq_len+1) format with lowercase support; older v1/v2
    # caches used a 1D overlapping-window stream and are NOT interchangeable.
    return f"tokens-{ds}-tok-{tok}-{case}-{sl}-{n}-v3"


def _tokens_artifact_ref(args) -> str:
    """Build the full `entity/project/name:alias` reference."""
    project = args.tokens_artifact_project or args.wandb_project
    entity  = args.tokens_artifact_entity  or args.wandb_entity
    name    = _tokens_artifact_name(args)
    if entity:
        return f"{entity}/{project}/{name}:latest"
    return f"{project}/{name}:latest"


def _try_download_tokens(args, cache_path: str) -> bool:
    """Master-only. Try fetching the tokens artifact and write to cache_path.

    Returns True on success, False otherwise (silently — caller falls back
    to local tokenization).
    """
    if not HAS_WANDB or args.no_tokens_artifact or args.no_wandb:
        return False
    ref = _tokens_artifact_ref(args)
    print(f"Looking for cached tokens artifact: {ref}", flush=True)
    try:
        api = wandb.Api()
        artifact = api.artifact(ref, type="dataset")
        download_dir = artifact.download()
        # Find the .pt file inside the artifact directory.
        files = [os.path.join(download_dir, f)
                 for f in os.listdir(download_dir) if f.endswith(".pt")]
        if not files:
            print(f"  ↳ artifact has no .pt file; falling back to local tokenize",
                  flush=True)
            return False
        # Atomic move into the expected cache path.
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        os.replace(files[0], cache_path)
        size_mb = os.path.getsize(cache_path) / 1024**2
        print(f"  ↳ downloaded {size_mb:.1f} MB → {cache_path}", flush=True)
        return True
    except Exception as e:
        print(f"  ↳ download skipped ({type(e).__name__}: {e})", flush=True)
        return False


def _try_upload_tokens(args, cache_path: str):
    """Master-only. Upload tokens cache to wandb as a versioned artifact.

    Idempotent: wandb dedupes by content hash, so a re-upload of the same
    file does not create a new version. Requires an active wandb run.
    """
    if not HAS_WANDB or args.no_tokens_artifact or args.no_wandb:
        return
    if wandb.run is None:
        print("  ↳ skip tokens upload (no active wandb run)", flush=True)
        return
    if not os.path.exists(cache_path):
        print(f"  ↳ skip tokens upload ({cache_path} missing)", flush=True)
        return
    name = _tokens_artifact_name(args)
    try:
        size_mb = os.path.getsize(cache_path) / 1024**2
        print(f"Uploading tokens artifact: {name}  ({size_mb:.1f} MB)...", flush=True)
        artifact = wandb.Artifact(
            name=name,
            type="dataset",
            description=f"Tokenized {args.dataset_name} via {args.tokenizer_name}",
            metadata={
                "dataset_name":   args.dataset_name,
                "dataset_split":  args.dataset_split,
                "tokenizer_name": args.tokenizer_name,
                "text_column":    args.text_column,
                "max_stories":    args.max_stories,
                "size_mb":        round(size_mb, 1),
            },
        )
        artifact.add_file(cache_path)
        wandb.log_artifact(artifact, aliases=["latest"])
        print(f"  ↳ uploaded (alias: latest)", flush=True)
    except Exception as e:
        print(f"  ↳ upload failed ({type(e).__name__}: {e}); training continues",
              flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1/2 metric helpers
# ─────────────────────────────────────────────────────────────────────────────

_LOG2_E = 1.0 / math.log(2.0)   # nats → bits multiplier


def _bits_per_byte(loss_nats: float, bytes_per_token: float) -> float:
    """Convert per-token cross-entropy (nats) to per-byte bits.

    bpb = (nats/token) × (bits/nat) × (token/byte)
        = loss_nats × log2(e) / bytes_per_token

    Universal cross-tokenizer axis — different vocabularies all collapse
    onto the same scale.
    """
    if bytes_per_token <= 0 or not math.isfinite(loss_nats):
        return float("nan")
    return loss_nats * _LOG2_E / bytes_per_token


def _mfu(flops_per_sec: float, peak_flops_per_device: float, world_size: int) -> float:
    """Model FLOPs Utilization — actual / theoretical-peak.

    Returns NaN when peak is unset (0). On v4-8: world_size=8, peak per device
    is per TensorCore. On v6e-N: peak per device is per chip.
    """
    if peak_flops_per_device <= 0:
        return float("nan")
    total_peak = peak_flops_per_device * max(world_size, 1)
    return flops_per_sec / total_peak


def _param_norm_l2(model) -> torch.Tensor:
    """Total L2 norm of all model parameters (on-device scalar)."""
    return torch.sqrt(
        torch.stack([
            p.detach().float().pow(2).sum()
            for p in model.parameters()
            if p.requires_grad
        ]).sum()
    )


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Mean top-k accuracy over (B*L) positions.

    logits:  (B*L, V)
    targets: (B*L,)
    Returns scalar in [0, 1].
    """
    topk = logits.topk(k, dim=-1).indices              # (B*L, k)
    return (topk == targets.unsqueeze(-1)).any(-1).float().mean()


def _precompile_step_graph(model, device, log) -> float:
    """Trigger XLA compilation of `model.step()` at startup.

    The end-of-epoch sample-generation path uses `model.step()`, which is a
    DIFFERENT XLA HLO from `model.forward()` and so doesn't share the
    in-process compilation cache built up by training. The first time the
    sampler hits `model.step()` is mid-training, and on torch_xla 2.9 (where
    the persistent disk cache is broken — pytorch/xla#9094) this causes a
    visible ~30s stall.

    Compiling a single step() up front amortises that into startup time,
    where the user is already waiting for the script to begin.

    Returns the wall-time the compile took (for logging).
    """
    import time as _time
    t0 = _time.time()
    with torch.no_grad():
        was_training = model.training
        model.eval()
        try:
            dummy = torch.zeros(1, dtype=torch.long, device=device)
            caches = model.allocate_inference_cache(
                batch_size=1, dtype=torch.float32, device=device,
            )
            logits, _ = model.step(dummy, caches)
            # Force materialisation so the compile actually fires.
            if HAS_XLA:
                _xla_sync()
                # One on-device read forces the compiled program to execute,
                # ensuring the cache entry is fully realised before training.
                _ = float(logits.float().sum().cpu())
            else:
                _ = float(logits.float().sum().cpu())
        finally:
            if was_training:
                model.train()
    return _time.time() - t0


def _gumbel_top_k_sample(logits: torch.Tensor, *,
                         temperature: float, top_k: int) -> torch.Tensor:
    """Categorical sample from logits — XLA-safe.

    Equivalent to `torch.multinomial(F.softmax(logits / T), num_samples=1)`
    but uses the Gumbel-max trick instead, so it lowers to pure XLA ops
    (rand_like / log / argmax) — no CPU fallback, no host-device sync.

        argmax(z + Gumbel(0,1)) ≡ Categorical(softmax(z))

    where Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1).
    """
    logits = logits / max(temperature, 1e-6)
    if top_k and 0 < top_k < logits.shape[-1]:
        v, _ = torch.topk(logits, k=top_k, dim=-1)
        logits = torch.where(
            logits < v[..., -1:],
            torch.full_like(logits, float("-inf")),
            logits,
        )
    u = torch.rand_like(logits).clamp_(min=1e-9)         # avoid log(0)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)


@torch.no_grad()
def _generate_sample(model, tokenizer, prompt: str, *,
                     max_new_tokens: int, temperature: float, top_k: int,
                     device, eos_id: int):
    """Autoregressive sample via Gumbel-max — TPU/XLA-safe.

    Two changes vs. the naïve loop that hung on TPU:
      * Gumbel-max instead of `torch.multinomial` — no CPU fallback.
      * No per-token `.item()` / EOS check inside the loop — generates a
        fixed `max_new_tokens`, collects on-device, and does ONE `.cpu()`
        transfer at the end. EOS truncation happens post-hoc on the host.

    Tradeoff: always pays for `max_new_tokens` worth of decode steps even
    if EOS arrives early. Fine for end-of-epoch monitoring; not what you
    want in a serving loop.

    Master-only; the caller already enforces the `is_master` guard.
    """
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [eos_id]
    prompt_tensor = torch.tensor(ids, dtype=torch.long, device=device)

    # Allocate per-layer caches.
    caches = model.allocate_inference_cache(
        batch_size=1, dtype=torch.float32, device=device,
    )

    # Prefill: feed each prompt token through model.step (single fixed-shape graph).
    for tok_id in prompt_tensor:
        _, caches = model.step(tok_id.unsqueeze(0), caches)

    # Generate exactly `max_new_tokens`. No early stop, no host syncs.
    new_tokens = []
    next_token = prompt_tensor[-1].unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, caches = model.step(next_token, caches)
        next_token = _gumbel_top_k_sample(
            logits, temperature=temperature, top_k=top_k,
        )
        new_tokens.append(next_token)

    # ONE host transfer for all generated tokens. `next_token` is shape (1,)
    # so stacking gives (max_new_tokens, 1) → squeeze to a flat list.
    out_ids = torch.stack(new_tokens, dim=0).squeeze(-1).cpu().tolist()

    # Post-hoc EOS truncation on the host side.
    if eos_id in out_ids:
        out_ids = out_ids[: out_ids.index(eos_id)]

    model.train()
    return tokenizer.decode(ids + out_ids, skip_special_tokens=False)


# ─────────────────────────────────────────────────────────────────────────────
# Inference benchmark (Mamba's selling point: O(1) decode + tiny state)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _benchmark_inference(model, cfg, device, *,
                         prefill_lens=(128, 512, 1024),
                         decode_tokens: int = 128,
                         batch_size: int = 1) -> dict:
    """Measure prefill throughput, decode throughput, and inference state size.

    The Mamba selling-point vs the SimpleStories transformer baseline is
    O(1) per-token decode (the SSM state is fixed-size; no growing KV cache)
    and tiny inference memory (constant in sequence length). This function
    quantifies both, with results comparable across model architectures.

    Numbers are single-stream (batch=1) — the realistic chat-serving case.
    Under SPMD the work runs replicated across all devices; reported
    latency reflects what one device sees, which is the right metric for
    serving one user's stream.

    Returns a flat dict with keys:
        inference/prefill_L{N}_tps           tokens/sec for an N-token prefill
        inference/prefill_L{N}_latency_ms    end-to-end prefill latency
        inference/decode_tps                 autoregressive tokens/sec
        inference/decode_latency_ms          mean per-token decode latency
        inference/state_bytes                inference state per batch slot
        inference/state_bytes_per_layer      same, divided by n_layer

    Mamba state size is `n_layer × B × d_model × (kernel_size + d_state) × 4`
    (fp32 SSM state + fp32 conv state). For comparison, a transformer's KV
    cache is `n_layer × 2 × B × n_kv × d_head × seq_len × dtype_bytes` —
    grows linearly in seq_len. Mamba's stays constant; that's the win.
    """
    was_training = model.training
    model.eval()
    results: dict = {}

    def _sync():
        if HAS_XLA:
            _xla_sync()
            xm.wait_device_ops()   # still the canonical "block until idle" call

    # ── Prefill throughput at multiple sequence lengths ────────────────────
    for L in prefill_lens:
        if L > cfg.seq_len:
            continue
        x = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        # Warmup (compiles the graph if it's the first time at this shape).
        _ = model(x); _sync()
        # Time multiple iters for a stable estimate.
        n_iter = 5
        t0 = time.time()
        for _ in range(n_iter):
            _ = model(x)
        _sync()
        elapsed = time.time() - t0
        tps = (batch_size * L * n_iter) / max(elapsed, 1e-9)
        per_call_ms = (elapsed / n_iter) * 1000.0
        results[f"inference/prefill_L{L}_tps"] = tps
        results[f"inference/prefill_L{L}_latency_ms"] = per_call_ms

    # ── Decode throughput (autoregressive O(1) step) ───────────────────────
    caches = model.allocate_inference_cache(
        batch_size=batch_size, dtype=torch.float32, device=device,
    )
    next_token = torch.zeros(batch_size, dtype=torch.long, device=device)
    # Warmup so the step graph is compiled (or cache-hit if pre-compile ran).
    _, caches = model.step(next_token, caches); _sync()
    t0 = time.time()
    for _ in range(decode_tokens):
        _, caches = model.step(next_token, caches)
    _sync()
    elapsed = time.time() - t0
    decode_tps = (batch_size * decode_tokens) / max(elapsed, 1e-9)
    decode_latency_ms = (elapsed / decode_tokens) * 1000.0
    results["inference/decode_tps"] = decode_tps
    results["inference/decode_latency_ms"] = decode_latency_ms

    # ── Inference state size (analytic; doesn't depend on seq_len) ────────
    # Per layer: conv_state (B, d_model, kernel_size) + ssm_state (B, d_model, d_state)
    # Both stored fp32 in our impl (4 bytes/elem).
    bytes_per_layer = batch_size * cfg.d_model * (cfg.kernel_size + cfg.d_state) * 4
    total_state_bytes = bytes_per_layer * cfg.n_layer
    results["inference/state_bytes"] = total_state_bytes
    results["inference/state_bytes_per_layer"] = bytes_per_layer
    results["inference/state_kb"] = total_state_bytes / 1024.0

    if was_training:
        model.train()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FLOP accounting (Mamba-exact analytic estimate)
# ─────────────────────────────────────────────────────────────────────────────

def _mamba_flops_per_token_forward(cfg: MambaLMConfig) -> dict:
    """Analytic forward FLOPs/token for our Mamba LM, decomposed by component.

    Convention: 1 FMA (fused multiply-add) = 2 FLOPs. A linear layer
    `Wx + b` with W ∈ ℝ^{out×in} costs `2 × in × out` FLOPs per output token.

    SSM-specific terms — NOT captured by the 6N approximation:
        * discretize:  dA = exp(δ·A)            → 2·D·N FLOPs (mul + exp ≈ 2 ops)
                       dBu = δ · B · x          → 2·D·N FLOPs (two mults per (d,n))
        * scan step:   h ← dA·h + dBu           → 2·D·N FLOPs
        * output:      y = ⟨C, h⟩_n + D·x       → 2·D·N + D FLOPs

    Conv1d is depthwise (groups=D), so only `2·K·D` per token, not 2·K·D².
    RMSNorm ≈ 3·D_in (sq, mean, rsqrt-mul).

    Backward is conventionally 2× forward (same as the 6N rule's 4N back vs 2N fwd),
    so training FLOPs = 3 × the value returned here.
    """
    D_in, D_m = cfg.d_input, cfg.d_model
    N         = cfg.d_state
    R         = cfg.dt_rank
    K         = cfg.kernel_size
    L         = cfg.n_layer
    V         = cfg.vocab_size

    # Per-block components ──────────────────────────────────────────────────
    proj_flops = 2 * (
        D_in * D_m            # input_proj
        + D_in * D_m          # res_proj
        + D_m * D_in          # output_proj
        + D_m * N             # x_B_proj
        + D_m * N             # x_C_proj
        + D_m * R             # x_dt_proj
        + R   * D_m           # dt_proj
    )
    conv_flops = 2 * K * D_m
    ssm_flops  = (
        2 * D_m * N           # dA = exp(δ·A): mul + exp
        + 2 * D_m * N         # dBu = δ·B·x: two mults per (d,n)
        + 2 * D_m * N         # h ← dA·h + dBu: mul + add
        + 2 * D_m * N         # y += sum_n C·h: mul + accumulate
        + D_m                 # D·x feedthrough
    )
    norm_flops = 3 * D_in     # RMSNorm in ResidualBlock

    per_block = proj_flops + conv_flops + ssm_flops + norm_flops
    blocks    = L * per_block

    final_norm = 3 * D_in
    lm_head    = 2 * D_in * V

    total_fwd  = blocks + final_norm + lm_head

    return {
        "proj_per_block":   proj_flops,
        "conv_per_block":   conv_flops,
        "ssm_per_block":    ssm_flops,
        "norm_per_block":   norm_flops,
        "per_block_total":  per_block,
        "blocks_total":     blocks,
        "final_norm":       final_norm,
        "lm_head":          lm_head,
        "forward_total":    total_fwd,
        "training_total":   3 * total_fwd,    # fwd + 2× bwd
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_and_cache(args, cache_path: str):
    """Tokenize the HF SimpleStories corpus into packed (N, seq_len+1) chunks.

    Matches the SimpleStories reference pipeline
    (`simple_stories_train.dataloaders.tokenize_and_concatenate`):
      * Optionally lowercases text (their `to_lower=True` flag)
      * Parallel batched tokenization via `datasets.map(num_proc=...)`
      * Concatenates story-by-story separated by EOS
      * Packs into non-overlapping (N, seq_len+1) chunks — `chunk[:-1]` is the
        input and `chunk[1:]` is the next-token target. Each token contributes
        to exactly one training example, so the standard
        `tokens_seen = num_steps × B × seq_len × world_size` accounting is
        accurate (no overlap inflation).

    Only the *master* rank should call this; other ranks load from cache.
    """
    if not HAS_DATASETS:
        raise RuntimeError(
            "The `datasets` package is required. Install with: pip install datasets"
        )
    if not HAS_TRANSFORMERS:
        raise RuntimeError(
            "The `transformers` package is required. Install with: pip install transformers"
        )

    print(f"Loading dataset {args.dataset_name!r}...", flush=True)
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    # Project to just the text column. SimpleStories ships several metadata
    # columns (topic, style, etc.) that we never read; dropping them frees
    # the unused arrays from the mmap'd Arrow table.
    all_cols = list(ds.column_names)
    if args.text_column not in all_cols:
        raise KeyError(
            f"text_column={args.text_column!r} not in dataset columns {all_cols!r}"
        )
    dropped_cols = [c for c in all_cols if c != args.text_column]
    if dropped_cols:
        ds = ds.select_columns([args.text_column])
        print(f"  ↳ kept {args.text_column!r}; dropped {len(dropped_cols)} unused: "
              f"{dropped_cols}", flush=True)
    if args.max_stories:
        ds = ds.select(range(min(args.max_stories, len(ds))))
    print(f"  ↳ {len(ds):,} stories", flush=True)

    print(f"Loading tokenizer {args.tokenizer_name!r}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    eos = tok.eos_token_id
    if eos is None:
        # Some tokenizers (e.g. plain BPE) lack an EOS — use 0 or the pad id.
        eos = tok.pad_token_id if tok.pad_token_id is not None else 0
        print(f"  ↳ tokenizer has no EOS; using id={eos} as separator")
    vocab_size = len(tok)                  # includes any added special tokens

    text_col  = args.text_column
    lowercase = bool(args.lowercase)

    # Batched parallel tokenization. The batched-with-num_proc path uses the
    # fast Rust tokenizers backend internally and pegs ~10× higher throughput
    # than the per-story Python loop. Returns one ids list per story (with EOS
    # appended) plus its UTF-8 byte count for the bits-per-byte axis.
    def _tok_fn(examples):
        texts = examples[text_col]
        if lowercase:
            texts = [t.lower() for t in texts]
        nbytes = [len(t.encode("utf-8")) for t in texts]
        encoded = tok(texts, add_special_tokens=False)["input_ids"]
        return {
            "ids":    [seq + [eos] for seq in encoded],
            "nbytes": nbytes,
        }

    n_proc = max(1, int(getattr(args, "tokenize_num_proc", 10)))
    print(f"Tokenizing {len(ds):,} stories  "
          f"(lowercase={lowercase}, num_proc={n_proc})...", flush=True)
    ds_tok = ds.map(
        _tok_fn,
        batched=True,
        batch_size=1000,
        num_proc=n_proc,
        remove_columns=ds.column_names,
        desc="tokenize",
    )

    # Concatenate into a single 1D stream + accumulate UTF-8 byte total.
    print("Concatenating story streams...", flush=True)
    stream: list[int] = []
    total_bytes = 0
    for ex in tqdm(ds_tok, desc="concatenate", unit="story"):
        stream.extend(ex["ids"])
        total_bytes += ex["nbytes"]
    n_tokens = len(stream)
    bytes_per_token = total_bytes / max(n_tokens, 1)
    print(f"  ↳ {n_tokens:,} tokens  (vocab={vocab_size}, "
          f"{bytes_per_token:.3f} bytes/token)", flush=True)

    # Pack into non-overlapping (N_chunks, seq_len+1) shape. The +1 is the
    # shift target — input = chunk[:-1], target = chunk[1:].
    chunk_size = args.seq_len + 1
    n_chunks = n_tokens // chunk_size
    if n_chunks < 2:
        raise RuntimeError(
            f"Only {n_chunks} chunks of size {chunk_size} from {n_tokens} tokens — "
            f"corpus too small or seq_len too large."
        )
    packed = torch.tensor(stream[: n_chunks * chunk_size], dtype=torch.int32)
    packed = packed.view(n_chunks, chunk_size)
    dropped_tail = n_tokens - n_chunks * chunk_size
    print(f"  ↳ packed: ({n_chunks:,}, {chunk_size}) = "
          f"{n_chunks * chunk_size:,} tokens  (dropped {dropped_tail} tail)",
          flush=True)

    # Train/val split at chunk granularity so the held-out chunks never
    # overlap with training context windows.
    n_val   = max(1, int(args.val_fraction * n_chunks))
    n_train = n_chunks - n_val
    train, val = packed[:n_train], packed[n_train:]

    cache_obj = {
        "train":           train,                  # (n_train, seq_len+1) int32
        "val":             val,                    # (n_val,   seq_len+1) int32
        "vocab_size":      vocab_size,
        "eos_id":          eos,
        "bytes_per_token": bytes_per_token,
        "total_bytes":     total_bytes,
        "tokenizer":       args.tokenizer_name,
        "dataset":         args.dataset_name,
        "lowercase":       lowercase,
        "seq_len":         args.seq_len,
        "schema_version":  3,                      # 3 = packed 2D + lowercase
    }
    tmp_path = cache_path + ".tmp"
    torch.save(cache_obj, tmp_path)
    os.replace(tmp_path, cache_path)        # atomic publish
    print(f"Cached tokenized corpus → {cache_path} "
          f"({os.path.getsize(cache_path) / 1024**2:.1f} MB)", flush=True)
    return cache_obj


def prepare_dataset(args, is_master: bool, world_size: int):
    """Load the tokenized corpus, in priority order:
        1. local cache file (--tokenized_cache)
        2. wandb artifact (downloaded into the cache file path)
        3. tokenize from scratch and write to the cache file

    SPMD-safe: single Python process — no inter-process rendezvous needed.
    `is_master` and `world_size` are accepted for API stability but ignored.

    Returns: (train, val, vocab_size, eos_id, bytes_per_token, freshly_tokenized)
    `freshly_tokenized` tells the caller whether the cache was just produced
    locally (and therefore needs uploading to wandb after `wandb.init`).
    """
    cache_path = args.tokenized_cache
    freshly_tokenized = False

    if not os.path.exists(cache_path) and is_master:
        # Try wandb artifact download before falling back to local tokenization.
        if not _try_download_tokens(args, cache_path):
            _tokenize_and_cache(args, cache_path)
            freshly_tokenized = True

    # No cross-process rendezvous needed — single Python process under SPMD.
    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
    schema = obj.get("schema_version", 1)
    if schema != 3:
        raise RuntimeError(
            f"Cache {cache_path} has schema_version={schema}; this script expects "
            f"v3 (packed 2D format with lowercase support). Delete the cache file "
            f"and re-run to regenerate."
        )
    cached_seq_len = obj.get("seq_len", None)
    if cached_seq_len is not None and cached_seq_len != args.seq_len:
        raise RuntimeError(
            f"Cache {cache_path} was tokenized with seq_len={cached_seq_len} but "
            f"this run requested --seq_len {args.seq_len}. Delete the cache or "
            f"pass --seq_len {cached_seq_len}."
        )
    cached_lowercase = obj.get("lowercase", None)
    if cached_lowercase is not None and cached_lowercase != bool(args.lowercase):
        raise RuntimeError(
            f"Cache {cache_path} was tokenized with lowercase={cached_lowercase} "
            f"but this run requested --lowercase={args.lowercase}. Delete the "
            f"cache or flip the flag to match."
        )
    bytes_per_token = obj.get("bytes_per_token", None)
    return (obj["train"], obj["val"], obj["vocab_size"], obj["eos_id"],
            bytes_per_token, freshly_tokenized)


class PackedTokenDataset(Dataset):
    """Non-overlapping packed chunks of shape (N, seq_len+1).

    Each item is the standard next-token-prediction pair derived by shifting
    the chunk by one: `x = chunk[:-1]`, `y = chunk[1:]`, both of length
    `seq_len`. One token contributes to exactly one training example, so
    `tokens_seen = num_steps × B × seq_len × world_size` reflects real data
    exposure (no overlap inflation as in the previous TokenStreamDataset).

    Matches the convention used by nanoGPT / llm.c / simple_stories_train,
    so token-budget comparisons against the SimpleStories baseline are 1:1.
    """

    def __init__(self, data: torch.Tensor):
        assert data.dtype == torch.int32 and data.ndim == 2, \
            f"expected (N, seq_len+1) int32; got {tuple(data.shape)} {data.dtype}"
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args, rank: int = 0):
    """Entry point — single Python process under SPMD (or single device).

    With `--multi_device`: SPMD is enabled in main() before this is called,
    a 1D data-parallel mesh is built here, and the global batch is sharded
    along the data dim across all available XLA devices. Single-process
    execution means the in-process XLA compilation cache is shared by all
    cores; there is no per-rank Python interpreter overhead.
    """

    # ── Device & master ────────────────────────────────────────────────────
    if HAS_XLA:
        device      = _xla_device()
        is_master   = _xla_is_master()      # True under SPMD (single proc)
        # Under SPMD, _xla_world_size() == 1 (one Python process), but the
        # *effective* parallelism is num_devices. Use num_devices everywhere
        # token accounting / scheduling needs the parallelism factor.
        spmd_active = HAS_SPMD and bool(getattr(args, "multi_device", False)) \
                      and (xr.is_spmd() if hasattr(xr, "is_spmd") else True)
        if spmd_active:
            num_devices = xr.global_runtime_device_count()
        else:
            num_devices = _xla_world_size()
        world_size = num_devices
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True
        world_size = 1
        num_devices = 1
        spmd_active = False

    def log(msg):
        if is_master:
            print(msg, flush=True)

    # Build the SPMD mesh once and reuse for every batch's mark_sharding call.
    spmd_mesh = None
    if spmd_active:
        import numpy as _np
        device_ids = _np.arange(num_devices)
        spmd_mesh = xs.Mesh(device_ids, (num_devices,), ("data",))
        log(f"SPMD enabled — mesh ('data',) over {num_devices} XLA devices")

    # ── wandb: master-only initialisation guard ────────────────────────────
    # Resolve once and cache; everything else just checks `wandb_enabled`.
    wandb_enabled = (
        is_master
        and HAS_WANDB
        and not getattr(args, "no_wandb", False)
        and getattr(args, "wandb_mode", "online") != "disabled"
    )
    if is_master and not HAS_WANDB and not getattr(args, "no_wandb", False):
        log("wandb not installed; continuing without wandb logging "
            "(pip install wandb to enable)")

    log(f"Device: {device}  (xla={HAS_XLA}, world_size={world_size}, rank={rank})")

    # Reproducibility — different seed per rank so data shuffles diverge.
    torch.manual_seed(args.seed + rank)

    # ── Data ───────────────────────────────────────────────────────────────
    (train_data, val_data, vocab_size, eos_id,
     bytes_per_token, freshly_tokenized) = prepare_dataset(
        args, is_master=is_master, world_size=world_size,
    )
    if bytes_per_token is None:
        # Old cache without byte stats: estimate from a sample of train_data.
        # Won't be perfectly accurate but lets bits/byte stay populated.
        bytes_per_token = 4.0    # SimpleStories rough average — overridden once retokenised
        log(f"WARNING: cache lacks bytes_per_token; assuming {bytes_per_token}. "
            f"Delete the cache to retokenise for accurate bits/byte.")
    # Packed-format size logging: report both chunk count (==len(dataset)) and
    # the underlying token count, so the per-epoch token budget is obvious.
    train_tok = train_data.shape[0] * train_data.shape[1]
    val_tok   = val_data.shape[0]   * val_data.shape[1]
    log(f"Chunks: train={train_data.shape[0]:,} ({train_tok:,} tok)  "
        f"val={val_data.shape[0]:,} ({val_tok:,} tok)  "
        f"vocab={vocab_size}  eos={eos_id}  bytes/token={bytes_per_token:.3f}")

    train_ds = PackedTokenDataset(train_data)
    val_ds   = PackedTokenDataset(val_data)

    # Under SPMD the DataLoader produces a GLOBAL batch (per-device × N) and
    # the GSPMD partitioner shards along the data dim — no DistributedSampler
    # required (single Python process, single iterator). args.batch_size is
    # interpreted as the per-device batch so token accounting (which uses
    # world_size = num_devices) stays unchanged from the xmp.spawn era.
    global_batch = args.batch_size * num_devices

    train_loader = DataLoader(
        train_ds, batch_size=global_batch,
        shuffle=True,
        drop_last=True, num_workers=args.num_workers,
        # pin_memory only helps CPU→CUDA PCIe transfers; on TPU the path is
        # CPU→TPU via infeed DMA and pinning has no effect (just wastes host
        # RAM). Disable on XLA, keep on for CUDA dev.
        pin_memory=not HAS_XLA,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=global_batch,
        shuffle=False, drop_last=True,
        num_workers=args.num_workers,
        # pin_memory only helps CPU→CUDA PCIe transfers; on TPU the path is
        # CPU→TPU via infeed DMA and pinning has no effect (just wastes host
        # RAM). Disable on XLA, keep on for CUDA dev.
        pin_memory=not HAS_XLA,
        persistent_workers=args.num_workers > 0,
    )

    # ── Model (sized to ~5M params for SimpleStories-5M parity) ────────────
    cfg = MambaLMConfig(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        d_input=args.d_input,
        d_model=args.d_model,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
        kernel_size=args.kernel_size,
        seq_len=args.seq_len,
        use_checkpoint=args.checkpoint,
    )
    model = MambaLMHeadModel(cfg).to(device)

    # ── Re-tie the lm_head to the embedding AFTER .to(device) ──────────────
    # In `MambaLMHeadModel.__init__`, when `tie_embeddings=True`, we do
    # `self.lm_head.weight = self.embedding.weight` so both Parameter slots
    # point to the same underlying tensor. But `model.to(device)` calls
    # `_apply` on each submodule independently and creates a fresh device-
    # side Parameter for each occurrence — without that memoisation working
    # reliably across all torch_xla / PyTorch versions, the tying breaks
    # silently and lm_head and embedding diverge during training.
    #
    # Symptom of unfixed: train converges to a "bigram-only" identity-map
    # model (forward-loss ~ 9 = uniform-baseline-ish, no sequence learning,
    # generations look superficially coherent due to top-k bigram sampling).
    # Run mamba-5m-ameevykj at v19 hit this exact failure — embedding.weight
    # std drifted to 0.042, lm_head.weight std to 0.090, mean diff 0.075.
    if cfg.tie_embeddings:
        model.lm_head.weight = model.embedding.weight
        assert (model.lm_head.weight.data_ptr()
                == model.embedding.weight.data_ptr()), (
            "tied-embedding aliasing broken after .to(device). "
            "Training would silently produce a degenerate bigram model."
        )

    num_params_unique     = model.num_parameters(unique=True)
    num_params_non_embed  = num_params_unique - model.embedding.weight.numel()
    log(f"Model: {num_params_unique:,} unique params  "
        f"(padded vocab={cfg.vocab_size})")
    log(f"Config: n_layer={cfg.n_layer}  d_input={cfg.d_input}  "
        f"d_model={cfg.d_model}  d_state={cfg.d_state}  seq_len={cfg.seq_len}")

    # ── Mamba-exact FLOP estimate (analytic, per token) ────────────────────
    flops_breakdown = _mamba_flops_per_token_forward(cfg)
    flops_fwd_per_token   = flops_breakdown["forward_total"]      # forward only
    flops_train_per_token = flops_breakdown["training_total"]     # 3× forward (fwd+bwd)
    log(f"FLOPs/token (Mamba-exact, fwd):    {flops_fwd_per_token:>14,}")
    log(f"FLOPs/token (Mamba-exact, train):  {flops_train_per_token:>14,}")
    log(f"FLOPs/token (6N, train approx):    {6 * num_params_non_embed:>14,}")

    # ── wandb.init (master only, after model so we can log param count) ────
    if wandb_enabled:
        wandb_config = {
            **vars(args),
            **{f"cfg.{k}": v for k, v in cfg.__dict__.items()},
            "num_parameters_unique":          num_params_unique,
            "num_parameters_non_embedding":   num_params_non_embed,
            # Both FLOP estimates exposed in the run config so every chart can
            # reference them and the comparison vs the transformer baseline is
            # reproducible from the run page alone.
            "flops_per_token_fwd_exact":      flops_fwd_per_token,
            "flops_per_token_train_exact":    flops_train_per_token,
            "flops_per_token_train_6N":       6 * num_params_non_embed,
            "flops_breakdown":                flops_breakdown,
            "world_size":                     world_size,
            "has_xla":                        HAS_XLA,
            "device":                         str(device),
        }
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                tags=list(args.wandb_tags) if args.wandb_tags else None,
                config=wandb_config,
            )
            # Architecture as a wandb summary (text). Avoid wandb.watch(): it
            # adds gradient hooks that fire .item()-style host-device sync.
            try:
                wandb.run.summary["model_architecture"] = str(model)
            except Exception:
                pass
            log(f"wandb run: {wandb.run.name}  ({wandb.run.url})")
        except Exception as e:
            log(f"wandb.init failed ({e!r}); disabling wandb for this run")
            wandb_enabled = False

    # ── Upload tokens artifact if we just tokenized locally (master only) ──
    # Idempotent: wandb dedupes by content hash, so re-uploading the same
    # bytes does not create a new version. Runs only after wandb.init so
    # the artifact is attached to this run's lineage.
    if freshly_tokenized and is_master and wandb_enabled:
        _try_upload_tokens(args, args.tokenized_cache)

    # ── Optimiser & schedule ───────────────────────────────────────────────
    optimizer = SyncfreeAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    tokens_per_step = args.batch_size * args.seq_len * world_size
    # Budget resolution priority: --max_tokens > --max_steps > --epochs.
    # `--max_tokens` is the cleanest knob for matching the SimpleStories
    # baseline (60k iter × 64 × 512 ≈ 2B tokens, regardless of how many
    # TPU cores you spread across).
    if args.max_tokens > 0:
        total_steps = max(1, args.max_tokens // tokens_per_step)
        budget_src  = (f"--max_tokens {args.max_tokens:,} "
                       f"({total_steps * tokens_per_step:,} tok actual)")
    elif args.max_steps > 0:
        total_steps = args.max_steps
        budget_src  = f"--max_steps {args.max_steps}"
    else:
        total_steps = args.epochs * steps_per_epoch
        budget_src  = f"--epochs {args.epochs}  ({steps_per_epoch} step/epoch)"

    # Warmup + cosine. eta_min=lr/10 matches the SimpleStories
    # `learning_rate_decay_frac=0.1` (cosine never falls below 10% of peak).
    if args.warmup_steps > 0:
        # Linear warmup (start at 1% of peak so step-0 isn't an exact zero
        # which would zero the param updates and confuse on-device norms).
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0,
            total_iters=args.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - args.warmup_steps, 1),
            eta_min=args.lr / 10.0,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=args.lr / 10.0,
        )

    log(f"Training: {budget_src}")
    log(f"  ↳ total_steps={total_steps}  warmup={args.warmup_steps}  "
        f"steps_per_epoch={steps_per_epoch}  ({tokens_per_step:,} tok / global step)")

    # ── DataLoader wrapping (async TPU prefetch) ───────────────────────────
    if HAS_XLA:
        train_iter_loader = pl.MpDeviceLoader(
            train_loader, device, device_prefetch_size=args.prefetch,
        )
        val_iter_loader = pl.MpDeviceLoader(
            val_loader, device, device_prefetch_size=args.prefetch,
        )
    else:
        train_iter_loader = train_loader
        val_iter_loader   = val_loader

    pad_vocab = cfg.vocab_size

    # ── Optional bf16 autocast ─────────────────────────────────────────────
    autocast_ctx = (
        torch.autocast(device_type=("xla" if HAS_XLA else device.type),
                       dtype=torch.bfloat16)
        if args.bf16 else _nullcontext()
    )

    # ── Resume (optional) ──────────────────────────────────────────────────
    # Must happen AFTER optimizer + scheduler exist (load_state_dict targets
    # them) and BEFORE the training loop reads global_step / best_val.
    resumed = _resume_from_checkpoint(
        args, model, optimizer, scheduler,
        device=device, is_master=is_master, world_size=world_size, log=log,
    )
    if resumed is not None:
        global_step  = resumed["global_step"]
        best_val     = resumed["best_val"]
        best_val_ppl = resumed["best_val_ppl"]
        # Derive start_epoch from global_step (intermediate checkpoints
        # don't carry an `epoch` field, but the integer division is always
        # correct provided steps_per_epoch matches the saved run — already
        # enforced by the seq_len/batch_size match check).
        start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        if start_epoch >= args.epochs:
            log(f"WARNING: resumed at step {global_step} (epoch {start_epoch + 1}) "
                f"but --epochs {args.epochs} leaves no remaining epochs. "
                f"Bump --epochs to continue training.")
    else:
        global_step  = 0
        best_val     = float("inf")
        best_val_ppl = float("inf")
        start_epoch  = 0

    # ── Pre-compile model.step (sample-generation graph) ───────────────────
    # The end-of-epoch sampler uses model.step(), a different XLA HLO than
    # model.forward(). Without this, the first sample at end-of-epoch-1
    # eats a ~30s cold-compile stall mid-training (visible as a z>15 spike
    # in throughput). Pre-compiling moves that cost into startup.
    if args.sample_every_n_epochs > 0:
        log("Pre-compiling model.step graph (sample-generation path)...")
        try:
            elapsed = _precompile_step_graph(model, device, log)
            log(f"  ↳ pre-compile complete in {elapsed:.1f}s")
        except Exception as e:
            log(f"  ↳ pre-compile failed ({e!r}); first sample-gen will compile mid-run")

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    log_buf = []
    t_start = time.time()
    t_last_log = t_start
    crashed = False
    # Track most recent train avg loss for the val/overfit_gap metric.
    last_train_loss_avg = float("nan")
    # Lazily-loaded tokenizer for end-of-epoch sample generation.
    sample_tok = None

    try:
        for epoch in range(start_epoch, args.epochs):
            # Per-epoch shuffle reproducibility — DataLoader's shuffle uses
            # the global torch RNG; reseed per epoch so resumed runs see
            # the same per-epoch ordering they would have seen continuously.
            torch.manual_seed(args.seed + epoch * 1009 + rank)

            # Mid-epoch resume: skip already-completed batches in the first
            # resumed epoch. islice consumes the iterator without invoking
            # the model — tokens are loaded but not trained on. Cheap.
            iter_loader = train_iter_loader
            if epoch == start_epoch and resumed is not None and steps_per_epoch > 0:
                skip_batches = global_step - epoch * steps_per_epoch
                if skip_batches > 0:
                    log(f"Resuming mid-epoch: skipping {skip_batches} "
                        f"already-completed batches in epoch {epoch + 1}.")
                    import itertools as _it
                    iter_loader = _it.islice(train_iter_loader, skip_batches, None)

            for x, y in iter_loader:
                if not HAS_XLA:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                x = x.long()                    # nn.Embedding wants int64
                y = y.long()

                # SPMD: shard the global batch along the data dim across
                # all XLA devices. Sequence dim is replicated. The GSPMD
                # partitioner takes care of per-device kernel launches.
                if spmd_mesh is not None:
                    xs.mark_sharding(x, spmd_mesh, ("data", None))
                    xs.mark_sharding(y, spmd_mesh, ("data", None))

                with autocast_ctx:
                    logits = model(x)                        # (B, L, V)
                    # cross_entropy is autocast-promoted to fp32 internally on XLA
                    loss = F.cross_entropy(
                        logits.view(-1, pad_vocab),
                        y.view(-1),
                        reduction="mean",
                    )

                # On-device NaN guard (§9.3) — no .item() needed.
                is_finite = torch.isfinite(loss)
                safe_loss = torch.where(is_finite, loss, torch.zeros_like(loss))

                optimizer.zero_grad(set_to_none=True)
                safe_loss.backward()

                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip,
                )

                if HAS_XLA:
                    if spmd_active:
                        # Under SPMD, gradients on replicated params are
                        # already aggregated by GSPMD during backward —
                        # `xm.optimizer_step`'s built-in all_reduce_gradients
                        # would be a redundant collective. Use plain step +
                        # explicit sync to keep the HLO clean.
                        optimizer.step()
                        _xla_sync()
                    else:
                        # xmp.spawn-style multi-process (now unused) or single
                        # XLA device — keep the canonical consolidated call.
                        xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                scheduler.step()

                # Diagnostics computed EVERY step (don't conditionalise — would
                # produce two distinct XLA graphs; see §9.4). Perplexity and
                # param_norm are computed on-device — they're cheap enough that
                # always-on costs less than two compiled graph variants.
                current_lr = torch.tensor(
                    scheduler.get_last_lr()[0], device=device, dtype=torch.float32,
                )
                safe_loss_d = safe_loss.detach().float()
                ppl_d       = torch.exp(torch.clamp(safe_loss_d, max=20.0))
                param_norm  = _param_norm_l2(model)         # on-device scalar
                log_buf.append(torch.stack([
                    safe_loss_d,
                    grad_norm.detach().float(),
                    (~is_finite).float().detach(),
                    current_lr,
                    ppl_d,
                    param_norm,
                ]))

                global_step += 1

                # Flush at log interval — one .cpu() across all buffered scalars.
                if global_step % args.log_interval == 0:
                    stacked = torch.stack(log_buf, dim=0)
                    values  = stacked.cpu().tolist()     # ONE host transfer
                    log_buf = []

                    if is_master:
                        avg_loss = sum(v[0] for v in values) / len(values)
                        last     = values[-1]
                        now      = time.time()
                        elapsed  = now - t_start
                        delta    = now - t_last_log
                        t_last_log = now
                        tok_done = global_step * args.batch_size * args.seq_len * world_size
                        # Instantaneous throughput (tokens processed in the
                        # just-completed log_interval window). Use this — NOT
                        # the cumulative tok_done/elapsed — as the primary
                        # perf metric, otherwise startup overhead (XLA compile,
                        # first-batch warmup, dataloader fill) pollutes every
                        # subsequent reading and you see a misleading
                        # logarithmic ramp instead of a steady-state value.
                        tok_window      = args.log_interval * args.batch_size * args.seq_len * world_size
                        tok_per_sec     = tok_window / max(delta, 1e-6)
                        tok_per_sec_avg = tok_done / max(elapsed, 1e-6)
                        step_time_ms    = (delta / max(args.log_interval, 1)) * 1000.0
                        print(
                            f"epoch {epoch+1}/{args.epochs}  "
                            f"step {global_step:>6d}/{total_steps}  "
                            f"loss {avg_loss:.4f} (last {last[0]:.4f}, "
                            f"ppl {math.exp(min(last[0], 20)):.2f})  "
                            f"|grad| {last[1]:.3f}  nan {int(last[2])}  "
                            f"lr {last[3]:.2e}  "
                            f"{tok_per_sec / 1000:.1f}k tok/s "
                            f"(avg {tok_per_sec_avg / 1000:.1f}k)",
                            flush=True,
                        )

                        # Track for val/overfit_gap regardless of wandb.
                        last_train_loss_avg = avg_loss

                        if wandb_enabled:
                            # Cumulative training-budget metrics.
                            flops_seen_6n     = 6.0 * num_params_non_embed * tok_done
                            flops_seen_exact  = flops_train_per_token * tok_done
                            tokens_per_param  = tok_done / max(num_params_non_embed, 1)
                            # Tier-1 metrics
                            bpb_train         = _bits_per_byte(avg_loss, bytes_per_token)
                            flops_per_sec_exact_inst = flops_train_per_token * tok_per_sec
                            mfu_inst          = _mfu(flops_per_sec_exact_inst,
                                                     args.peak_flops_per_device, world_size)
                            # Tier-2 metrics — pull param_norm out of the on-device stack
                            param_norm_v      = last[5]
                            update_norm_ratio = (last[3] * last[1] / max(param_norm_v, 1e-12))
                            try:
                                wandb.log(
                                    {
                                        # ── Loss family ──
                                        "train/loss":              last[0],
                                        "train/loss_avg_window":   avg_loss,
                                        "train/perplexity":        last[4],
                                        "train/loss_bits_per_byte": bpb_train,
                                        # ── Optimiser dynamics ──
                                        "train/grad_norm":         last[1],
                                        "train/param_norm_l2":     param_norm_v,
                                        "train/update_norm_ratio": update_norm_ratio,
                                        "train/nan_count":         int(last[2]),
                                        "train/lr":                last[3],
                                        # ── Step counters ──
                                        "train/global_step":       global_step,
                                        "train/epoch":             epoch + 1,
                                        "train/tokens_seen":       tok_done,
                                        "train/tokens_per_param":  tokens_per_param,
                                        "train/flops_seen":        flops_seen_6n,
                                        "train/flops_seen_exact":  flops_seen_exact,
                                        # ── Instantaneous perf ──
                                        "perf/tokens_per_sec":         tok_per_sec,
                                        "perf/flops_per_sec":          6.0 * num_params_non_embed * tok_per_sec,
                                        "perf/flops_per_sec_exact":    flops_per_sec_exact_inst,
                                        "perf/mfu":                    mfu_inst,
                                        "perf/step_time_ms":           step_time_ms,
                                        # ── Cumulative averages (whole-run) ──
                                        "perf/tokens_per_sec_avg":     tok_per_sec_avg,
                                        "perf/flops_per_sec_avg":      6.0 * num_params_non_embed * tok_per_sec_avg,
                                        "perf/flops_per_sec_avg_exact": flops_train_per_token * tok_per_sec_avg,
                                        "perf/wall_clock_seconds":     elapsed,
                                    },
                                    step=global_step,
                                )
                            except Exception as e:
                                # Don't crash training over a logging hiccup.
                                print(f"  ↳ wandb.log failed: {e!r}", flush=True)

                # ── Intermediate validation (sweep early-termination signal) ──
                # Single-process under SPMD; evaluate() reports the per-device
                # mean (an unbiased estimator of the global mean for iid data).
                # Lightweight: --val_every_n_batches batches.
                if (args.val_every_n_steps > 0
                        and global_step % args.val_every_n_steps == 0):
                    eval_out = evaluate(
                        model, val_iter_loader, pad_vocab, device,
                        max_batches=args.val_every_n_batches,
                        autocast_ctx=autocast_ctx,
                        spmd_mesh=spmd_mesh, num_devices=num_devices,
                    )
                    iv_loss = eval_out["loss"]
                    iv_ppl  = (math.exp(min(iv_loss, 20))
                               if math.isfinite(iv_loss) else float("nan"))
                    iv_bpb  = _bits_per_byte(iv_loss, bytes_per_token)
                    iv_new_best = iv_loss < best_val
                    if iv_new_best:
                        best_val     = iv_loss
                        best_val_ppl = iv_ppl
                    if is_master:
                        log(f"  ↳ step {global_step}  val_loss {iv_loss:.4f}  "
                            f"val_ppl {iv_ppl:.2f}  bpb {iv_bpb:.4f}  "
                            f"(best {best_val:.4f})")
                        if wandb_enabled:
                            try:
                                wandb.log({
                                    "val/loss":               iv_loss,
                                    "val/perplexity":         iv_ppl,
                                    "val/loss_bits_per_byte": iv_bpb,
                                    "val/best_loss":          best_val,
                                    "val/best_perplexity":    best_val_ppl,
                                }, step=global_step)
                            except Exception as e:
                                print(f"  ↳ wandb.log (iv) failed: {e!r}",
                                      flush=True)
                        # Snapshot every intermediate eval — wandb dedupes
                        # identical content but distinct training states each
                        # produce a new artifact version.
                        _checkpoint_after_eval(
                            model=model, cfg=cfg, args=args,
                            save_path=args.save_path,
                            step=global_step, val_loss=iv_loss,
                            new_best=iv_new_best,
                            wandb_enabled=wandb_enabled,
                            optimizer=optimizer, scheduler=scheduler,
                            best_val=best_val, best_val_ppl=best_val_ppl,
                            extra_aliases=["intermediate"],
                            extra_metadata={"loss_bits_per_byte": iv_bpb},
                        )

                # ── Budget early-stop ──
                # Honour --max_tokens / --max_steps without finishing the epoch.
                # Force one final val pass so the run never exits without a
                # val/loss readout (matters for sweep ranking).
                if global_step >= total_steps:
                    if (args.val_every_n_steps == 0
                            or global_step % args.val_every_n_steps != 0):
                        eval_out = evaluate(
                            model, val_iter_loader, pad_vocab, device,
                            max_batches=args.eval_batches,
                            autocast_ctx=autocast_ctx,
                            spmd_mesh=spmd_mesh, num_devices=num_devices,
                        )
                        fv_loss = eval_out["loss"]
                        fv_ppl  = (math.exp(min(fv_loss, 20))
                                   if math.isfinite(fv_loss) else float("nan"))
                        fv_bpb  = _bits_per_byte(fv_loss, bytes_per_token)
                        fv_new_best = fv_loss < best_val
                        if fv_new_best:
                            best_val     = fv_loss
                            best_val_ppl = fv_ppl
                        if is_master:
                            log(f"  ↳ final val (budget hit)  loss {fv_loss:.4f}  "
                                f"ppl {fv_ppl:.2f}  bpb {fv_bpb:.4f}")
                            if wandb_enabled:
                                try:
                                    wandb.log({
                                        "val/loss":               fv_loss,
                                        "val/perplexity":         fv_ppl,
                                        "val/loss_bits_per_byte": fv_bpb,
                                        "val/best_loss":          best_val,
                                        "val/best_perplexity":    best_val_ppl,
                                    }, step=global_step)
                                except Exception as e:
                                    print(f"  ↳ wandb.log (final-iv) failed: {e!r}",
                                          flush=True)
                            _checkpoint_after_eval(
                                model=model, cfg=cfg, args=args,
                                save_path=args.save_path,
                                step=global_step, val_loss=fv_loss,
                                new_best=fv_new_best,
                                wandb_enabled=wandb_enabled,
                                optimizer=optimizer, scheduler=scheduler,
                                best_val=best_val, best_val_ppl=best_val_ppl,
                                extra_aliases=["budget-hit"],
                                extra_metadata={"loss_bits_per_byte": fv_bpb},
                            )
                    break

            # Outer epoch loop — propagate the early-stop break.
            if global_step >= total_steps:
                log(f"Reached step budget ({global_step}/{total_steps}); "
                    f"stopping epoch loop.")
                break

            # ── End-of-epoch validation ────────────────────────────────────
            eval_out = evaluate(model, val_iter_loader, pad_vocab,
                                device, max_batches=args.eval_batches,
                                autocast_ctx=autocast_ctx,
                                spmd_mesh=spmd_mesh, num_devices=num_devices)
            val_loss = eval_out["loss"]
            val_top1 = eval_out["top1"]
            val_top5 = eval_out["top5"]
            val_ppl  = math.exp(min(val_loss, 20)) if math.isfinite(val_loss) else float("nan")
            val_bpb  = _bits_per_byte(val_loss, bytes_per_token)
            overfit_gap = (val_loss - last_train_loss_avg
                           if math.isfinite(last_train_loss_avg) else float("nan"))
            log(f"─── epoch {epoch+1} done   val_loss {val_loss:.4f}   "
                f"val_ppl {val_ppl:.2f}   bpb {val_bpb:.4f}   "
                f"top1 {val_top1*100:.1f}%   top5 {val_top5*100:.1f}%   "
                f"overfit_gap {overfit_gap:+.4f} ───")

            # Snapshot every epoch — capture new_best BEFORE we update so the
            # checkpoint is correctly tagged.
            new_best = val_loss < best_val
            if new_best:
                best_val     = val_loss
                best_val_ppl = val_ppl
            if is_master:
                _checkpoint_after_eval(
                    model=model, cfg=cfg, args=args,
                    save_path=args.save_path,
                    step=global_step, val_loss=val_loss,
                    new_best=new_best,
                    wandb_enabled=wandb_enabled,
                    epoch=epoch + 1,
                    optimizer=optimizer, scheduler=scheduler,
                    best_val=best_val, best_val_ppl=best_val_ppl,
                    extra_aliases=["epoch-end"],
                    extra_metadata={
                        "loss_bits_per_byte": val_bpb,
                        "top1_accuracy":      val_top1,
                        "top5_accuracy":      val_top5,
                        "overfit_gap":        overfit_gap,
                    },
                )

            if wandb_enabled:
                tok_done_ep         = global_step * args.batch_size * args.seq_len * world_size
                flops_seen_6n_ep    = 6.0 * num_params_non_embed * tok_done_ep
                flops_seen_exact_ep = flops_train_per_token * tok_done_ep
                try:
                    wandb.log(
                        {
                            # ── Loss family ──
                            "val/loss":              val_loss,
                            "val/perplexity":        val_ppl,
                            "val/loss_bits_per_byte": val_bpb,
                            "val/best_loss":         best_val,
                            "val/best_perplexity":   best_val_ppl,
                            # ── Tier-1: train/val gap ──
                            "val/overfit_gap":       overfit_gap,
                            # ── Tier-2: accuracies ──
                            "val/top1_accuracy":     val_top1,
                            "val/top5_accuracy":     val_top5,
                            # ── Budget axes (so val/loss is plottable vs tokens) ──
                            "val/tokens_seen":       tok_done_ep,
                            "val/flops_seen":        flops_seen_6n_ep,
                            "val/flops_seen_exact":  flops_seen_exact_ep,
                            "epoch":                 epoch + 1,
                        },
                        step=global_step,
                    )
                except Exception as e:
                    print(f"  ↳ wandb.log (val) failed: {e!r}", flush=True)

            # ── Inference benchmark (decode/prefill throughput, state size) ──
            # Quantifies Mamba's main serving advantage vs the transformer
            # baseline: O(1) per-token decode and a fixed-size inference
            # state that doesn't grow with sequence length.
            if (is_master
                    and args.inference_bench_every_n_epochs > 0
                    and (epoch + 1) % args.inference_bench_every_n_epochs == 0):
                try:
                    bench = _benchmark_inference(model, cfg, device)
                    # Compact log line — full metrics go to wandb.
                    decode_tps   = bench.get("inference/decode_tps", float("nan"))
                    decode_lat   = bench.get("inference/decode_latency_ms", float("nan"))
                    state_kb     = bench.get("inference/state_kb", float("nan"))
                    prefill_keys = sorted(k for k in bench if "prefill_L" in k and "tps" in k)
                    prefill_str  = "  ".join(
                        f"{k.split('_L')[1].split('_')[0]}={bench[k]/1e3:.1f}k"
                        for k in prefill_keys
                    )
                    log(f"  📊 inference: decode {decode_tps:.0f} tok/s "
                        f"({decode_lat:.2f}ms/tok)  prefill[{prefill_str}]  "
                        f"state {state_kb:.1f} KB/stream")
                    if wandb_enabled:
                        try:
                            wandb.log(bench, step=global_step)
                        except Exception as e:
                            print(f"  ↳ wandb.log (inference) failed: {e!r}",
                                  flush=True)
                except Exception as e:
                    log(f"  ↳ inference benchmark failed: {e!r}")

            # ── Sample text generation (master only, end-of-epoch) ─────────
            if (is_master and args.sample_every_n_epochs > 0
                    and (epoch + 1) % args.sample_every_n_epochs == 0
                    and HAS_TRANSFORMERS):
                if sample_tok is None:
                    try:
                        sample_tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
                    except Exception as e:
                        log(f"  ↳ failed to load tokenizer for sampling: {e!r}")
                        sample_tok = False    # don't try again
                if sample_tok:
                    samples = []
                    for prompt in args.sample_prompts:
                        try:
                            txt = _generate_sample(
                                model, sample_tok, prompt,
                                max_new_tokens=args.sample_max_new_tokens,
                                temperature=args.sample_temperature,
                                top_k=args.sample_top_k,
                                device=device, eos_id=eos_id,
                            )
                            samples.append((prompt, txt))
                        except Exception as e:
                            log(f"  ↳ sample for {prompt!r} failed: {e!r}")
                    if samples:
                        for prompt, txt in samples:
                            log(f"  ✏  [{prompt!r}] → {txt!r}")
                        if wandb_enabled:
                            try:
                                table = wandb.Table(columns=["epoch", "prompt", "generation"])
                                for prompt, txt in samples:
                                    table.add_data(epoch + 1, prompt, txt)
                                wandb.log({"samples/text": table}, step=global_step)
                            except Exception as e:
                                print(f"  ↳ wandb sample table log failed: {e!r}", flush=True)

        # ── Final checkpoint (master only) ─────────────────────────────────
        if args.save_path and is_master:
            final_path = args.save_path.replace(".pt", "_final.pt")
            _save_checkpoint(
                model, cfg, args, final_path,
                extra={"val_loss": best_val, "epoch": args.epochs},
                wandb_enabled=wandb_enabled,
                artifact_aliases=["final"],
                optimizer=optimizer, scheduler=scheduler,
                global_step=global_step, epoch=args.epochs,
                best_val=best_val, best_val_ppl=best_val_ppl,
            )

        # ── XLA debug summary (master only) ────────────────────────────────
        if HAS_XLA and is_master:
            print("\n" + met.short_metrics_report())

    except Exception:
        crashed = True
        raise
    finally:
        if wandb_enabled:
            try:
                wandb.finish(exit_code=1 if crashed else 0)
            except Exception:
                pass


def _state_to_cpu(obj):
    """Recursively migrate every torch.Tensor in a state-dict-shaped tree to CPU.

    Optimizer state dicts contain tensors (Adam's `m`, `v`, step counters) that
    live on the XLA device after step() has run; serialising them straight to
    disk would carry XLA tensor metadata that fails to load on a fresh process.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        coll = [_state_to_cpu(v) for v in obj]
        return type(obj)(coll)
    return obj


def _checkpoint_after_eval(*, model, cfg, args, save_path, step, val_loss,
                           new_best, wandb_enabled, epoch=None,
                           optimizer=None, scheduler=None,
                           best_val=None, best_val_ppl=None,
                           extra_aliases=None, extra_metadata=None):
    """Save + upload a checkpoint artifact tied to a just-completed evaluation.

    Called from every eval site (intermediate val, final-val-on-budget-hit,
    end-of-epoch val) so that wandb has a model snapshot for each evaluated
    point in training. Aliases:
        * "latest"       — auto-applied by wandb to the newest version
        * "step-{N}"     — every save (lets you fetch a specific step)
        * "epoch-{N}"    — when an `epoch` is provided
        * "best"         — when this eval set a new best val_loss
    Caller must hold the master-only guard (no SPMD-replicated I/O).

    The optimizer + scheduler + best counters are persisted so the run is
    fully resumable from this artifact via `--resume_from / --resume_artifact`.
    """
    if not (args.save_path and save_path):
        return
    if not math.isfinite(val_loss):
        # Avoid serialising a model with NaN params from a corrupted step.
        print(f"  ↳ skip checkpoint at step {step}: val_loss is non-finite",
              flush=True)
        return
    aliases = [f"step-{step}"]
    if epoch is not None:
        aliases.append(f"epoch-{epoch}")
    if new_best:
        aliases.append("best")
    if extra_aliases:
        aliases.extend(extra_aliases)
    extra = {"val_loss": float(val_loss), "step": int(step), "best": bool(new_best)}
    if epoch is not None:
        extra["epoch"] = int(epoch)
    if extra_metadata:
        extra.update(extra_metadata)
    _save_checkpoint(
        model, cfg, args, save_path,
        extra=extra,
        wandb_enabled=wandb_enabled,
        artifact_aliases=aliases,
        optimizer=optimizer, scheduler=scheduler,
        global_step=step, epoch=epoch,
        best_val=best_val, best_val_ppl=best_val_ppl,
    )


def _save_checkpoint(model, cfg, args, path, extra=None,
                     wandb_enabled: bool = False,
                     artifact_aliases=None,
                     *,
                     optimizer=None, scheduler=None,
                     global_step=None, epoch=None,
                     best_val=None, best_val_ppl=None):
    """Save checkpoint locally; optionally upload as a wandb artifact.

    Caller is responsible for the master-only guard. The wandb upload is
    wrapped in try/except so a failed upload never crashes training.

    When optimizer / scheduler / counters are passed, the payload is fully
    self-sufficient for resume — `_resume_from_checkpoint` rebuilds the run
    state from these fields and skips the already-completed batches.
    """
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "config":     cfg.__dict__,
        "state_dict": cpu_state,
        "args":       vars(args),
    }
    # Resume metadata. None values are intentionally elided so a checkpoint
    # without these fields stays loadable as model-only.
    if optimizer is not None:
        payload["optim_state"] = _state_to_cpu(optimizer.state_dict())
    if scheduler is not None:
        payload["sched_state"] = scheduler.state_dict()
    if global_step is not None:
        payload["global_step"] = int(global_step)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_val is not None:
        payload["best_val"] = float(best_val)
    if best_val_ppl is not None:
        payload["best_val_ppl"] = float(best_val_ppl)
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  ↳ checkpoint → {path}", flush=True)

    if wandb_enabled and HAS_WANDB and wandb.run is not None:
        try:
            run_id = wandb.run.id
            artifact_name = f"mamba-5m-{run_id}"
            metadata = {
                "path":   path,
                "config": cfg.__dict__,
            }
            if extra:
                metadata.update({k: v for k, v in extra.items()
                                 if isinstance(v, (int, float, str, bool))})
            artifact = wandb.Artifact(
                name=artifact_name, type="model", metadata=metadata,
            )
            artifact.add_file(path)
            aliases = list(artifact_aliases) if artifact_aliases else None
            wandb.log_artifact(artifact, aliases=aliases)
            print(f"  ↳ wandb artifact: {artifact_name}  aliases={aliases}",
                  flush=True)
        except Exception as e:
            print(f"  ↳ wandb artifact upload failed: {e!r}", flush=True)


_RESUME_LOCAL_PATH = "_resume_checkpoint.pt"


def _resume_from_checkpoint(args, model, optimizer, scheduler, *,
                            device, is_master, world_size, log) -> dict | None:
    """Restore training state from a local file or wandb artifact.

    Resolution order:
        1. --resume_artifact ENTITY/PROJECT/NAME:ALIAS  (master downloads,
           all ranks read from `_resume_checkpoint.pt`)
        2. --resume_from PATH                           (all ranks read PATH)

    Loads:
        * model weights              (always)
        * optimizer + scheduler      (unless --resume_reset_optimizer)
        * counters                   (global_step, epoch, best_val, best_val_ppl)

    Validates architecture and seq_len match before touching state. Mismatch
    fails loudly rather than silently corrupting training. Returns the run
    state dict or None if no resume was requested.

    SPMD-safe: single Python process, so the master-only download just
    runs once and the file is then read by the same process. No inter-
    process rendezvous needed.
    """
    cache_path = None

    if args.resume_artifact:
        if not HAS_WANDB:
            raise RuntimeError("--resume_artifact requires the wandb package")
        if is_master:
            log(f"Downloading checkpoint artifact: {args.resume_artifact}")
            try:
                api = wandb.Api()
                artifact = api.artifact(args.resume_artifact, type="model")
                d = artifact.download()
                files = [os.path.join(d, f)
                         for f in os.listdir(d) if f.endswith(".pt")]
                if not files:
                    raise RuntimeError(
                        f"Artifact {args.resume_artifact} contains no .pt file"
                    )
                os.replace(files[0], _RESUME_LOCAL_PATH)
                size_mb = os.path.getsize(_RESUME_LOCAL_PATH) / 1024**2
                log(f"  ↳ downloaded {size_mb:.1f} MB → {_RESUME_LOCAL_PATH}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download resume artifact "
                    f"{args.resume_artifact!r}: {e!r}"
                )
        cache_path = _RESUME_LOCAL_PATH
    elif args.resume_from:
        cache_path = args.resume_from
    else:
        return None

    # No cross-process rendezvous needed — single Python process under SPMD.
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Resume checkpoint {cache_path!r} does not exist on rank "
            f"{0 if is_master else '?'}."
        )

    log(f"Loading checkpoint from {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)

    # ── Architecture compatibility check ───────────────────────────────────
    saved_cfg = payload.get("config", {})
    cur_cfg   = model.config.__dict__
    arch_keys = ("n_layer", "d_input", "d_model", "d_state", "dt_rank",
                 "kernel_size", "vocab_size", "tie_embeddings")
    mismatches = [(k, saved_cfg.get(k), cur_cfg.get(k)) for k in arch_keys
                  if saved_cfg.get(k) != cur_cfg.get(k)]
    if mismatches:
        details = "; ".join(f"{k}: saved={s!r} current={c!r}"
                            for k, s, c in mismatches)
        raise RuntimeError(
            f"Architecture mismatch on resume — refuse to load. {details}"
        )

    # Sequence length AND per-device batch size must match. Both feed into
    # `steps_per_epoch`, which `start_epoch = global_step // steps_per_epoch`
    # uses to derive the resume offset. A mismatch silently mis-aligns
    # which batches get re-trained vs. skipped — a hard-to-debug
    # data-revisitation bug. Hard-fail with a fix-it message.
    saved_args = payload.get("args", {})
    seq_len_saved = saved_args.get("seq_len", args.seq_len)
    if seq_len_saved != args.seq_len:
        raise RuntimeError(
            f"seq_len mismatch on resume: saved={seq_len_saved} current={args.seq_len}. "
            f"steps_per_epoch and the LR schedule depend on seq_len."
        )
    batch_saved = saved_args.get("batch_size", args.batch_size)
    if batch_saved != args.batch_size:
        raise RuntimeError(
            f"batch_size mismatch on resume: saved={batch_saved} current={args.batch_size}. "
            f"steps_per_epoch = len(train_ds) // (batch_size × num_devices); "
            f"changing it mis-aligns the resume offset and re-trains already-completed "
            f"batches (or skips fresh ones). Pass --batch_size {batch_saved} or accept "
            f"a fresh run."
        )

    # ── Restore model weights ──────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(payload["state_dict"],
                                                strict=True)
    if missing or unexpected:
        log(f"  ↳ load_state_dict report — missing={missing} unexpected={unexpected}")
    model.to(device)

    # ── Restore optimizer + scheduler (or rebuild fresh) ───────────────────
    loaded_optim = False
    loaded_sched = False
    if not args.resume_reset_optimizer:
        if "optim_state" in payload:
            try:
                optimizer.load_state_dict(payload["optim_state"])
                loaded_optim = True
            except Exception as e:
                log(f"  ↳ optimizer load failed ({e!r}); continuing with fresh optimizer")
        if "sched_state" in payload:
            try:
                scheduler.load_state_dict(payload["sched_state"])
                loaded_sched = True
            except Exception as e:
                log(f"  ↳ scheduler load failed ({e!r}); continuing with fresh scheduler")

    state = {
        "global_step":  int(payload.get("global_step", 0)),
        "best_val":     float(payload.get("best_val",     float("inf"))),
        "best_val_ppl": float(payload.get("best_val_ppl", float("inf"))),
        "epoch":        int(payload.get("epoch", 0)),
        "loaded_optim": loaded_optim,
        "loaded_sched": loaded_sched,
    }
    log(f"  ↳ resumed: global_step={state['global_step']}  "
        f"epoch={state['epoch']}  best_val={state['best_val']:.4f}  "
        f"optim={'loaded' if loaded_optim else 'fresh'}  "
        f"sched={'loaded' if loaded_sched else 'fresh'}")
    return state


@torch.no_grad()
def evaluate(model, val_loader, pad_vocab, device,
             max_batches: int = 50, autocast_ctx=None,
             spmd_mesh=None, num_devices: int = 1) -> dict:
    """Mean cross-entropy + top-1 / top-5 accuracy over up to `max_batches`.

    Aggregation policy
    ------------------
    Under SPMD (single Python process, sharded data): each device's
    `cross_entropy(reduction='mean')` is the mean of its local data shard.
    For iid-shuffled test data this is already an UNBIASED ESTIMATOR of
    the global mean — the only thing we'd gain by aggregating is a
    sqrt(num_devices) variance reduction, which is negligible at our
    sample counts (thousands of tokens per device per batch).

    We previously tried `xm.all_reduce(SUM) / num_devices` here, on the
    advice of a code review that assumed `xm.all_reduce` works the same
    way under SPMD as under xmp.spawn. It does not. Under single-process
    SPMD, the per-device scalar is annotated as replicated and
    `xm.all_reduce` no-ops; the divide-by-N then silently shrunk the
    reported value (observed: 16× too small on TPU v4-4 — the model's
    actual loss was 9.07 but reports came out as 0.57). The 4× factor
    matches `num_devices`; a residual factor seems to come from how
    GSPMD treats the scalar reduction of a `cross_entropy(reduction='mean')`
    over a sharded input.

    Returns: {"loss": float, "top1": float, "top5": float}.
    """
    model.eval()
    losses, top1s, top5s = [], [], []
    autocast_ctx = autocast_ctx or _nullcontext()
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        if not HAS_XLA:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        x = x.long(); y = y.long()
        if spmd_mesh is not None:
            xs.mark_sharding(x, spmd_mesh, ("data", None))
            xs.mark_sharding(y, spmd_mesh, ("data", None))
        with autocast_ctx:
            logits = model(x)
            flat_logits = logits.view(-1, pad_vocab)
            flat_targets = y.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)
        losses.append(loss.float())
        top1s.append(_topk_accuracy(flat_logits.float(), flat_targets, k=1))
        top5s.append(_topk_accuracy(flat_logits.float(), flat_targets, k=5))
    if not losses:
        model.train()
        return {"loss": float("nan"), "top1": float("nan"), "top5": float("nan")}
    avg_loss = torch.stack(losses).mean()
    avg_top1 = torch.stack(top1s).mean()
    avg_top5 = torch.stack(top5s).mean()
    # Each device's per-shard mean is an unbiased estimator of the global
    # mean (iid shards). The earlier "explicit xm.all_reduce(SUM) / N"
    # pattern was incorrect under single-process SPMD: `xm.all_reduce` is a
    # no-op there (no inter-process collective), and the divide silently
    # made values num_devices× too small — observed empirically as a 4×
    # bias on TPU v4-4. Trust the per-device estimator; the variance gain
    # from cross-device aggregation is negligible at our sample counts.
    # SINGLE .cpu() across all three scalars (xla_tpu_reference §2.2).
    triplet = torch.stack([avg_loss, avg_top1, avg_top5]).cpu().tolist()
    model.train()
    return {"loss": triplet[0], "top1": triplet[1], "top5": triplet[2]}


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

class _nullcontext:
    def __enter__(self):  return None
    def __exit__(self, *a):  return False


# ─────────────────────────────────────────────────────────────────────────────
# YAML config loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml_config(path: str) -> dict:
    """Load a YAML config file. Normalises any dash-keys to underscore-keys
    so YAML can use either `grad-clip:` or `grad_clip:` interchangeably."""
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "--config requires PyYAML. Install with: pip install pyyaml"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config {path}: expected a YAML mapping at the top level")
    return {k.replace("-", "_"): v for k, v in cfg.items()}


def main():
    # ── Two-pass argparse for YAML support ─────────────────────────────────
    # First pass: discover --config (and nothing else) without erroring on
    # the rest. Second pass: build the full parser with YAML values applied
    # as defaults via parser.set_defaults(), then re-parse so explicit CLI
    # flags still win over YAML values.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Mamba LM training — SimpleStories-5M-equivalent on TPU",
    )
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file. Values become "
                             "defaults; explicit CLI flags override them.")

    # ── Data (HF SimpleStories defaults) ───────────────────────────────────
    parser.add_argument("--dataset_name",    default="lennart-finke/SimpleStories",
                        help="HF dataset path (e.g. 'lennart-finke/SimpleStories')")
    parser.add_argument("--dataset_split",   default="train")
    parser.add_argument("--text_column",     default="story",
                        help="Name of the text column in the HF dataset")
    parser.add_argument("--tokenizer_name",  default="SimpleStories/SimpleStories-5M",
                        help="HF tokenizer (same vocab as the baseline transformer)")
    parser.add_argument("--tokenized_cache", default="simplestories_tokens_v3.pt",
                        help="Local cache of tokenized corpus. v3 = packed (N, "
                             "seq_len+1) format with lowercase support; v1/v2 "
                             "files are NOT compatible — delete them or pass a "
                             "fresh path.")
    parser.add_argument("--max_stories",     type=int, default=0,
                        help="Limit corpus size for smoke tests (0 = full)")
    parser.add_argument("--val_fraction",    type=float, default=0.01)
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Lowercase text before tokenization "
                             "(matches SimpleStories `to_lower=True`). "
                             "Pass --no-lowercase to disable.")
    parser.add_argument("--tokenize_num_proc", type=int, default=10,
                        help="Parallel processes for datasets.map() "
                             "tokenization. 1 disables parallelism.")

    # ── Model (param-matched to SimpleStories-5M, canonical Mamba E=2) ─────
    parser.add_argument("--n_layer",     type=int, default=9)     # Mamba prefers depth
    parser.add_argument("--d_input",     type=int, default=256)   # = baseline d_model
    parser.add_argument("--d_model",     type=int, default=512)   # E=2 (paper standard)
    parser.add_argument("--d_state",     type=int, default=16)
    parser.add_argument("--dt_rank",     type=int, default=32)   # = d_model/16 (Mamba paper)
    parser.add_argument("--kernel_size", type=int, default=4)
    parser.add_argument("--seq_len",     type=int, default=512)   # = baseline n_ctx

    # ── Optimisation ───────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--batch_size",   type=int,   default=128)   # per device
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip",    type=float, default=2.0)
    parser.add_argument("--log_interval", type=int,   default=20)
    parser.add_argument("--eval_batches", type=int,   default=50)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--prefetch",     type=int,   default=4,
                        help="MpDeviceLoader prefetch depth")

    # ── Token / step budget (overrides --epochs when > 0) ──────────────────
    # `--max_tokens` is the cleanest way to match the SimpleStories baseline:
    # 60 000 iter × 64 batch × 512 ctx ≈ 2 000 000 000 tokens, regardless of
    # how many TPU cores you spread across (the script computes the right
    # step count given world_size).
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Stop after this many tokens (0 = use --epochs / "
                             "--max_steps). SimpleStories 5M baseline ≈ 2e9.")
    parser.add_argument("--max_steps",  type=int, default=0,
                        help="Stop after this many optimizer steps "
                             "(ignored if --max_tokens > 0). 0 disables.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear LR warmup steps before cosine decay. "
                             "SimpleStories 35M default = 600. 0 disables.")
    parser.add_argument("--val_every_n_steps", type=int, default=0,
                        help="Run intermediate validation every N steps. "
                             "0 = end-of-epoch only. Required for sweep "
                             "early-termination on val/loss.")
    parser.add_argument("--val_every_n_batches", type=int, default=20,
                        help="Batches per intermediate val pass "
                             "(end-of-epoch val uses --eval_batches).")

    # ── Memory & precision ─────────────────────────────────────────────────
    # BooleanOptionalAction so YAML can set `bf16: true|false` and CLI can
    # use `--bf16` or `--no-bf16` interchangeably.
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Gradient checkpointing on each ResidualBlock")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Forward in bfloat16 autocast (cross_entropy stays fp32)")

    # ── Hardware peak FLOPs (for MFU computation) ──────────────────────────
    # Defaults assume bf16 with fp32 accumulation. Per *logical XLA device*,
    # which on v4 is one TensorCore (half-chip = 137.5 TFLOPs/s peak) and on
    # v6e is one chip (~459 TFLOPs/s peak). Pass 0 to disable MFU logging.
    parser.add_argument("--peak_flops_per_device", type=float, default=0.0,
                        help="Peak FLOPs/s per logical XLA device. "
                             "TPU v4 TensorCore bf16 ≈ 1.375e14, "
                             "v6e chip bf16 ≈ 4.59e14. 0 disables MFU.")

    # ── Inference benchmarking at end of epoch (decode/prefill perf) ───────
    parser.add_argument("--inference_bench_every_n_epochs", type=int, default=1,
                        help="Run inference benchmark every N epochs (decode "
                             "tok/s, prefill tok/s, state size). 0 disables.")

    # ── Sample generation at end of epoch (qualitative monitoring) ─────────
    parser.add_argument("--sample_every_n_epochs", type=int, default=1,
                        help="Generate text samples every N epochs. 0 disables.")
    parser.add_argument("--sample_max_new_tokens", type=int, default=200)
    parser.add_argument("--sample_top_k",          type=int, default=40)
    parser.add_argument("--sample_temperature",    type=float, default=0.8)
    parser.add_argument("--sample_prompts", nargs="+", default=[
        "Once upon a time, ",
        "There was a little girl named",
        "The dog ran",
    ])

    # ── I/O & parallelism ──────────────────────────────────────────────────
    parser.add_argument("--save_path", default="mamba_simplestories_5m.pt")
    parser.add_argument("--multi_device", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable SPMD across all available XLA devices "
                             "(single Python process; GSPMD shards the global "
                             "batch). Off = single device.")

    # ── Resume training from a previous run ─────────────────────────────────
    # Two sources, in priority order: --resume_artifact (master pulls from
    # wandb, all ranks read locally) > --resume_from (all ranks read PATH).
    # Architecture / seq_len / vocab must match the checkpoint exactly.
    parser.add_argument("--resume_from", default=None,
                        help="Path to a local .pt checkpoint to resume from. "
                             "All ranks must see the same path.")
    parser.add_argument("--resume_artifact", default=None,
                        help="wandb artifact reference to resume from "
                             "(format: 'entity/project/name:alias'). Master "
                             "downloads, all ranks then read the local cache.")
    parser.add_argument("--resume_reset_optimizer", action="store_true",
                        help="Load model weights only — rebuild optimizer + "
                             "scheduler from scratch. Use to warm-start with "
                             "different lr / wd / schedule.")

    # ── wandb (master-only side-effects; gracefully optional) ──────────────
    parser.add_argument("--wandb_project", default="mamba-simplestories")
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--wandb_run_name", default=None,
                        help="If unset, wandb auto-generates a name")
    parser.add_argument("--wandb_mode",    default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_tags",    nargs="+", default=[])
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging entirely")

    # ── Tokenized-dataset caching via wandb artifacts ──────────────────────
    # Re-tokenizing the full SimpleStories corpus takes minutes; uploading
    # the tokenized cache to wandb once and pulling it on subsequent runs
    # turns startup into a single ~1 GB download.
    parser.add_argument("--tokens_artifact", default=None,
                        help="wandb Artifact name for cached tokens. Default: "
                             "auto-derived from --dataset_name + --tokenizer_name "
                             "+ --max_stories.")
    parser.add_argument("--tokens_artifact_project", default=None,
                        help="wandb project that hosts the tokens artifact. "
                             "Defaults to --wandb_project.")
    parser.add_argument("--tokens_artifact_entity", default=None,
                        help="wandb entity for the tokens artifact. "
                             "Defaults to --wandb_entity (or your default entity).")
    parser.add_argument("--no_tokens_artifact", action="store_true",
                        help="Skip both downloading and uploading the tokenized "
                             "cache as a wandb artifact (use local file only).")

    # Apply YAML defaults BEFORE the final parse so explicit CLI flags still
    # take priority over YAML values. Unknown keys are flagged loudly to
    # catch typos (otherwise a misspelled `lerning_rate:` would silently
    # do nothing).
    if pre_args.config:
        yaml_cfg = _load_yaml_config(pre_args.config)
        valid_dests = {a.dest for a in parser._actions}
        unknown = sorted(set(yaml_cfg) - valid_dests)
        if unknown:
            raise SystemExit(
                f"--config {pre_args.config}: unknown keys {unknown}. "
                f"Valid keys: {sorted(valid_dests)}"
            )
        parser.set_defaults(**yaml_cfg)
        print(f"Loaded config {pre_args.config}: "
              f"{len(yaml_cfg)} key(s) → {sorted(yaml_cfg)}", flush=True)

    args = parser.parse_args()

    if args.multi_device:
        if not HAS_XLA:
            raise RuntimeError("--multi_device requires torch_xla")
        if not HAS_SPMD:
            raise RuntimeError(
                "--multi_device now uses SPMD (torch_xla.distributed.spmd). "
                "Your torch_xla install lacks SPMD support — upgrade to "
                "torch_xla >= 2.0 (the project pins ~2.9 in requirements_tpu.txt)."
            )
        # SPMD MUST be enabled BEFORE any XLA device is acquired. The
        # `_xla_device()` call inside train() is the first acquisition;
        # toggling here is safe because no XLA device has been touched yet.
        xr.use_spmd()
        train(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
