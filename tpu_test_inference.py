"""
Inference / test-set evaluation for trained Mamba LMs.

Loads a wandb model artifact (or a local .pt checkpoint), runs the model on
the SimpleStories *test* split (the held-out partition the upstream baseline
also evaluates against), and reports a comprehensive set of inference and
quality metrics under the `test/` and `inference/` wandb namespaces.

What it measures
----------------
1. Test-set quality (matches SimpleStories' `train_llama.py` val-loop math):
       - test/loss              cross-entropy nats/token
       - test/perplexity        exp(loss)
       - test/loss_bits_per_byte    tokenizer-agnostic — direct cross-arch axis
       - test/top1_accuracy / test/top5_accuracy
   The full test split is iterated (no `val_max_steps` cap, unlike during
   training). Tokens are packed into non-overlapping `(N, seq_len+1)` chunks
   and shifted into `(x, y)` pairs the same way training does.

2. Serving-side performance (Mamba's selling point vs the transformer):
       - inference/decode_tps              autoregressive tokens/sec
       - inference/decode_latency_ms       per-token decode latency
       - inference/prefill_L{N}_tps        forward throughput at length N
       - inference/state_bytes / state_kb  inference state — CONSTANT in seq_len

3. Qualitative samples — Gumbel-max top-k generations from configurable
   prompts, logged as a wandb Table.

Methodology notes
-----------------
- Tokenisation pipeline mirrors `tpu_train.py` exactly (lowercase BPE via the
  SimpleStories-5M tokenizer, EOS-separated, packed). This is what makes the
  test-loss numbers comparable to your training run's val-loss numbers AND to
  the SimpleStories transformer baseline's val/test loss.
- Vocab + tokenizer match the upstream baseline so bits-per-byte is the
  apples-to-apples cross-architecture comparison axis.
- The `test/` namespace is reserved for THIS script's metrics so a single
  wandb project can hold both training (val/) and test (test/) runs without
  collision.

Usage
-----
    # From a local checkpoint
    python tpu_test_inference.py --checkpoint_path mamba_simplestories_5m_final.pt

    # From a wandb artifact
    python tpu_test_inference.py \
        --artifact entity/mamba-simplestories/mamba-5m-{run_id}:best

    # Multi-device on TPU v4-4
    PJRT_DEVICE=TPU python tpu_test_inference.py \
        --artifact ... --multi_device --bf16

    # YAML config (architecture inferred from checkpoint, but defaults match)
    python tpu_test_inference.py --config config_5M.yaml --artifact ...
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─── XLA imports (mirror tpu_train.py) ───────────────────────────────────────
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
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
except ImportError:
    HAS_XLA = False
    xm = pl = None  # type: ignore
    _xla_world_size = _xla_process_index = _xla_is_master = _xla_device = None  # type: ignore

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

# ─── Optional dependencies ───────────────────────────────────────────────────
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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # type: ignore

from mamba.mamba_llm_tpu import MambaLMHeadModel, MambaLMConfig

# Reuse helpers from tpu_train so the test-side numerics are identical to
# the training-side ones (BPB formula, top-k accuracy, sampler, inference
# bench, packed dataset, YAML loader, null context).
from tpu_train import (
    PackedTokenDataset,
    _benchmark_inference,
    _bits_per_byte,
    _generate_sample,
    _load_yaml_config,
    _nullcontext,
    _topk_accuracy,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test-set tokenisation (mirrors tpu_train._tokenize_and_cache for `test`)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_test_data(args, log) -> tuple[torch.Tensor, int, int, float]:
    """Tokenise the SimpleStories test split into `(N, seq_len+1)` chunks.

    Same recipe as training: lowercase → fast BPE via `datasets.map(num_proc=…)`
    → EOS-separated 1D stream → packed into non-overlapping fixed-length
    chunks. Packed chunk size is `seq_len+1` so `chunk[:-1]/chunk[1:]` gives
    the standard next-token shift pair without losing any tokens to overlap.
    Returns: (data, vocab_size, eos_id, bytes_per_token).
    """
    if not (HAS_DATASETS and HAS_TRANSFORMERS):
        raise RuntimeError(
            "Tokenisation needs `datasets` and `transformers`. "
            "Install with: pip install datasets transformers"
        )

    cache = args.tokenized_test_cache
    if cache and os.path.exists(cache):
        log(f"Loading cached test tokens from {cache}")
        obj = torch.load(cache, map_location="cpu", weights_only=False)
        if obj.get("seq_len") != args.seq_len:
            raise RuntimeError(
                f"Test cache {cache} was tokenised with seq_len={obj.get('seq_len')} "
                f"but this run requested seq_len={args.seq_len}. Delete the cache."
            )
        if obj.get("lowercase") != bool(args.lowercase):
            raise RuntimeError(
                f"Test cache lowercase={obj.get('lowercase')} != requested {args.lowercase}"
            )
        return (obj["data"], obj["vocab_size"], obj["eos_id"], obj["bytes_per_token"])

    log(f"Loading dataset {args.dataset_name!r} (split={args.dataset_split!r})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.text_column not in ds.column_names:
        raise KeyError(
            f"text_column={args.text_column!r} not in {ds.column_names!r}"
        )
    ds = ds.select_columns([args.text_column])
    if args.max_stories:
        ds = ds.select(range(min(args.max_stories, len(ds))))
    log(f"  ↳ {len(ds):,} test stories")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    eos = tok.eos_token_id if tok.eos_token_id is not None else (
          tok.pad_token_id if tok.pad_token_id is not None else 0)
    vocab_size = len(tok)
    text_col, lowercase = args.text_column, bool(args.lowercase)

    def _tok_fn(examples):
        texts = examples[text_col]
        if lowercase:
            texts = [t.lower() for t in texts]
        nbytes = [len(t.encode("utf-8")) for t in texts]
        encoded = tok(texts, add_special_tokens=False)["input_ids"]
        return {"ids": [seq + [eos] for seq in encoded], "nbytes": nbytes}

    n_proc = max(1, int(args.tokenize_num_proc))
    log(f"Tokenising {len(ds):,} test stories  "
        f"(lowercase={lowercase}, n_proc={n_proc})...")
    ds_tok = ds.map(
        _tok_fn, batched=True, batch_size=1000, num_proc=n_proc,
        remove_columns=ds.column_names, desc="tokenise",
    )

    log("Concatenating story streams...")
    stream: list[int] = []
    total_bytes = 0
    for ex in tqdm(ds_tok, desc="concatenate", unit="story"):
        stream.extend(ex["ids"])
        total_bytes += ex["nbytes"]
    n_tokens = len(stream)
    bytes_per_token = total_bytes / max(n_tokens, 1)
    log(f"  ↳ {n_tokens:,} tokens  (vocab={vocab_size}, "
        f"{bytes_per_token:.3f} bytes/token)")

    chunk_size = args.seq_len + 1
    n_chunks = n_tokens // chunk_size
    if n_chunks < 1:
        raise RuntimeError(
            f"Test data too small: only {n_chunks} chunks of size {chunk_size}"
        )
    data = torch.tensor(stream[: n_chunks * chunk_size], dtype=torch.int32)
    data = data.view(n_chunks, chunk_size)
    log(f"  ↳ packed: ({n_chunks:,}, {chunk_size}) = "
        f"{n_chunks * chunk_size:,} tokens "
        f"(dropped {n_tokens - n_chunks * chunk_size} tail)")

    if cache:
        tmp = cache + ".tmp"
        torch.save({"data": data, "vocab_size": vocab_size, "eos_id": eos,
                    "bytes_per_token": bytes_per_token, "total_bytes": total_bytes,
                    "seq_len": args.seq_len, "lowercase": lowercase,
                    "tokenizer": args.tokenizer_name,
                    "dataset": args.dataset_name, "split": args.dataset_split,
                    "schema_version": 3}, tmp)
        os.replace(tmp, cache)
        log(f"Cached → {cache} ({os.path.getsize(cache) / 1024**2:.1f} MB)")

    return data, vocab_size, eos, bytes_per_token


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint loading (local file or wandb artifact)
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint(args, log) -> tuple[dict, MambaLMConfig]:
    """Resolve --artifact (downloads via wandb.Api) or --checkpoint_path to
    a path, load the payload, and reconstruct the saved MambaLMConfig.
    """
    cache_path = args.checkpoint_path

    if args.artifact:
        if not HAS_WANDB:
            raise RuntimeError("--artifact requires the wandb package")
        log(f"Downloading artifact: {args.artifact}")
        try:
            api = wandb.Api()
            artifact = api.artifact(args.artifact, type="model")
            d = artifact.download()
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")]
            if not files:
                raise RuntimeError(
                    f"Artifact {args.artifact!r} contains no .pt file"
                )
            cache_path = files[0]
            size_mb = os.path.getsize(cache_path) / 1024**2
            log(f"  ↳ {size_mb:.1f} MB → {cache_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact {args.artifact!r}: {e!r}")

    if not cache_path:
        raise SystemExit("Pass --checkpoint_path PATH or --artifact ENTITY/PROJECT/NAME:ALIAS")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Checkpoint not found: {cache_path}")

    log(f"Loading checkpoint from {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "config" not in payload or "state_dict" not in payload:
        raise RuntimeError(
            f"Checkpoint {cache_path} missing 'config' or 'state_dict'. "
            f"Was it produced by tpu_train.py?"
        )
    cfg = MambaLMConfig(**payload["config"])
    return payload, cfg


# ═══════════════════════════════════════════════════════════════════════════
# Test-set evaluation — full pass, no cap
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test(model, test_loader, pad_vocab, device, *,
                  bytes_per_token, autocast_ctx=None,
                  spmd_mesh=None, num_devices: int = 1, log=print) -> dict:
    """Iterate the FULL test loader and report aggregate quality metrics.

    Mirrors `tpu_train.evaluate()` math exactly (so test/loss is on the same
    scale as val/loss from training), but without the `max_batches` cap —
    test reporting should use every available test token. Under SPMD, the
    per-device cross_entropy mean is reported as-is (it's an unbiased
    estimator of the global mean for iid-shuffled test data); see
    `tpu_train.evaluate()` for the rationale.
    """
    model.eval()
    losses, top1s, top5s = [], [], []
    autocast_ctx = autocast_ctx or _nullcontext()
    n_samples = 0
    n_tokens  = 0
    t0 = time.time()
    iterator = tqdm(test_loader, desc="test", unit="batch")
    for x, y in iterator:
        if not HAS_XLA:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        x = x.long(); y = y.long()
        if spmd_mesh is not None:
            xs.mark_sharding(x, spmd_mesh, ("data", None))
            xs.mark_sharding(y, spmd_mesh, ("data", None))
        n_samples += int(x.shape[0])
        n_tokens  += int(x.shape[0] * x.shape[1])
        with autocast_ctx:
            logits = model(x)
            flat_logits  = logits.view(-1, pad_vocab)
            flat_targets = y.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)
        losses.append(loss.float())
        top1s.append(_topk_accuracy(flat_logits.float(), flat_targets, k=1))
        top5s.append(_topk_accuracy(flat_logits.float(), flat_targets, k=5))
    elapsed = time.time() - t0

    if not losses:
        model.train()
        return {"test/loss": float("nan"), "test/perplexity": float("nan"),
                "test/loss_bits_per_byte": float("nan"),
                "test/top1_accuracy": float("nan"), "test/top5_accuracy": float("nan"),
                "test/n_samples": 0, "test/n_tokens": 0,
                "test/elapsed_seconds": elapsed, "test/throughput_tps": 0.0}

    avg_loss = torch.stack(losses).mean()
    avg_top1 = torch.stack(top1s).mean()
    avg_top5 = torch.stack(top5s).mean()
    # Each device's `F.cross_entropy(reduction='mean')` over its data shard is
    # already an unbiased estimator of the global mean (iid shards). We
    # previously tried `xm.all_reduce(REDUCE_SUM) / num_devices` here under
    # the assumption that GSPMD wouldn't auto-aggregate, but on single-process
    # SPMD `xm.all_reduce` is effectively a no-op (no inter-process collective
    # to invoke) and the divide silently made values num_devices× too small —
    # observed empirically as a 4× bias on TPU v4-4. Trust the per-device
    # estimator; variance over thousands of samples is negligible.
    triplet = torch.stack([avg_loss, avg_top1, avg_top5]).cpu().tolist()
    loss_nats, top1, top5 = triplet
    ppl = math.exp(min(loss_nats, 20)) if math.isfinite(loss_nats) else float("nan")
    bpb = _bits_per_byte(loss_nats, bytes_per_token)

    model.train()
    return {
        "test/loss":               loss_nats,
        "test/perplexity":         ppl,
        "test/loss_bits_per_byte": bpb,
        "test/top1_accuracy":      top1,
        "test/top5_accuracy":      top5,
        "test/n_samples":          n_samples,
        "test/n_tokens":           n_tokens,
        "test/elapsed_seconds":    elapsed,
        "test/throughput_tps":     n_tokens / max(elapsed, 1e-9),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI / main
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mamba inference test on the SimpleStories test split",
    )
    p.add_argument("--config", default=None,
                   help="YAML config (defaults; CLI overrides win)")

    # Checkpoint source. Note: NOT named `--checkpoint` to avoid colliding
    # with the training YAML's `checkpoint: true` (gradient-checkpointing
    # boolean). The YAML inherits cleanly when this is `--checkpoint_path`.
    p.add_argument("--checkpoint_path", default=None,
                   help="Local .pt checkpoint produced by tpu_train.py")
    p.add_argument("--artifact", default=None,
                   help="wandb artifact ref: entity/project/name:alias")

    # Test data
    p.add_argument("--dataset_name",        default="lennart-finke/SimpleStories")
    p.add_argument("--dataset_split",       default="test")
    p.add_argument("--text_column",         default="story")
    p.add_argument("--tokenizer_name",      default="SimpleStories/SimpleStories-5M")
    p.add_argument("--max_stories",         type=int, default=0)
    p.add_argument("--lowercase",           action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--tokenize_num_proc",   type=int, default=10)
    p.add_argument("--tokenized_test_cache", default="simplestories_test_tokens_v3.pt")

    # Inference / batching
    p.add_argument("--seq_len",     type=int, default=512,
                   help="Must match the checkpoint's seq_len.")
    p.add_argument("--batch_size",  type=int, default=128,
                   help="Per-device under SPMD; multiplied by num_devices for "
                        "the DataLoader.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch",    type=int, default=4)

    # Hardware
    p.add_argument("--multi_device", action=argparse.BooleanOptionalAction,
                   default=False, help="SPMD across all XLA devices")
    p.add_argument("--bf16",         action=argparse.BooleanOptionalAction,
                   default=False, help="bf16 autocast for forward")
    p.add_argument("--seed",         type=int, default=0)

    # Sample text generation
    p.add_argument("--sample_max_new_tokens", type=int, default=200)
    p.add_argument("--sample_top_k",          type=int, default=40)
    p.add_argument("--sample_temperature",    type=float, default=0.8)
    p.add_argument("--sample_prompts", nargs="+", default=[
        "Once upon a time, ",
        "There was a little girl named",
        "The dog ran",
        "In a small village,",
    ])
    p.add_argument("--no_samples", action=argparse.BooleanOptionalAction,
                   default=False, help="Skip sample generation entirely")

    # Inference benchmark sweep
    p.add_argument("--inference_decode_tokens", type=int, default=128,
                   help="How many decode steps to time")
    p.add_argument("--inference_prefill_lens", type=int, nargs="+",
                   default=[128, 512, 1024, 2048],
                   help="Sequence lengths to benchmark prefill at "
                        "(values > seq_len are skipped)")
    p.add_argument("--no_inference_bench", action=argparse.BooleanOptionalAction,
                   default=False, help="Skip the decode/prefill benchmark")

    # wandb
    p.add_argument("--wandb_project",  default="mamba-simplestories")
    p.add_argument("--wandb_entity",   default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode",     default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags",     nargs="+", default=["inference-test"])
    p.add_argument("--no_wandb",       action=argparse.BooleanOptionalAction,
                   default=False)
    return p


def main():
    # Two-pass YAML pre-discovery (mirrors tpu_train).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    parser = _build_parser()

    if pre_args.config:
        yaml_cfg = _load_yaml_config(pre_args.config)
        valid = {a.dest for a in parser._actions}
        unknown = sorted(set(yaml_cfg) - valid)
        if unknown:
            # Soft warning rather than hard fail — the training YAML has
            # extra keys (lr, weight_decay, warmup_steps, …) that are
            # irrelevant to inference but not "wrong" to inherit.
            print(f"Loaded config {pre_args.config}: ignoring training-only keys "
                  f"{unknown}", flush=True)
            for k in unknown:
                yaml_cfg.pop(k, None)
        parser.set_defaults(**yaml_cfg)
        print(f"Loaded config {pre_args.config}: applied "
              f"{len(yaml_cfg)} key(s)", flush=True)

    args = parser.parse_args()

    if not (args.checkpoint_path or args.artifact):
        raise SystemExit(
            "Pass --checkpoint_path PATH or --artifact ENTITY/PROJECT/NAME:ALIAS"
        )

    # ── Enable SPMD BEFORE acquiring an XLA device (same rule as tpu_train) ─
    if args.multi_device:
        if not HAS_XLA:
            raise RuntimeError("--multi_device requires torch_xla")
        if not HAS_SPMD:
            raise RuntimeError(
                "--multi_device requires SPMD support (torch_xla 2.0+). "
                "Upgrade torch_xla."
            )
        xr.use_spmd()

    # ── Device + parallelism ───────────────────────────────────────────────
    if HAS_XLA:
        device      = _xla_device()
        is_master   = _xla_is_master()
        spmd_active = HAS_SPMD and bool(args.multi_device) \
                      and (xr.is_spmd() if hasattr(xr, "is_spmd") else True)
        num_devices = (xr.global_runtime_device_count()
                       if spmd_active else _xla_world_size())
    else:
        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master   = True
        spmd_active = False
        num_devices = 1

    def log(msg: str):
        if is_master:
            print(msg, flush=True)

    log(f"Device: {device}  (xla={HAS_XLA}, spmd={spmd_active}, "
        f"num_devices={num_devices})")
    torch.manual_seed(args.seed)

    spmd_mesh = None
    if spmd_active:
        import numpy as _np
        spmd_mesh = xs.Mesh(_np.arange(num_devices), (num_devices,), ("data",))
        log(f"SPMD enabled — mesh ('data',) over {num_devices} XLA devices")

    # ── wandb init ─────────────────────────────────────────────────────────
    wandb_enabled = (
        is_master and HAS_WANDB and not args.no_wandb
        and args.wandb_mode != "disabled"
    )
    if wandb_enabled:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                tags=list(args.wandb_tags) if args.wandb_tags else None,
                config={**vars(args), "stage": "inference-test"},
                job_type="inference",
            )
            log(f"wandb run: {wandb.run.name}  ({wandb.run.url})")
        except Exception as e:
            log(f"wandb.init failed ({e!r}); continuing without wandb")
            wandb_enabled = False

    # ── Load checkpoint + reconstruct model ────────────────────────────────
    payload, cfg = load_checkpoint(args, log)
    log(f"Model config: n_layer={cfg.n_layer}  d_input={cfg.d_input}  "
        f"d_model={cfg.d_model}  d_state={cfg.d_state}  dt_rank={cfg.dt_rank}  "
        f"seq_len={cfg.seq_len}  vocab_size={cfg.vocab_size}")
    if cfg.seq_len != args.seq_len:
        log(f"  ↳ overriding --seq_len {args.seq_len} → {cfg.seq_len} (from checkpoint)")
        args.seq_len = cfg.seq_len

    model = MambaLMHeadModel(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    n_params_unique = model.num_parameters(unique=True)
    n_params_non_embed = n_params_unique - model.embedding.weight.numel()
    log(f"Loaded model: {n_params_unique:,} unique params "
        f"({n_params_non_embed:,} non-embedding); checkpoint metadata: "
        f"step={payload.get('global_step')} epoch={payload.get('epoch')} "
        f"val_loss={payload.get('val_loss')}")

    if wandb_enabled:
        wandb.run.summary.update({
            "model/n_parameters_unique":      n_params_unique,
            "model/n_parameters_non_embed":   n_params_non_embed,
            "model/cfg":                      cfg.__dict__,
            "checkpoint/global_step":         payload.get("global_step"),
            "checkpoint/epoch":               payload.get("epoch"),
            "checkpoint/best_val":            payload.get("best_val"),
            "checkpoint/val_loss":            payload.get("val_loss"),
            "checkpoint/source":              args.artifact or args.checkpoint_path,
        })

    # ── Test data ──────────────────────────────────────────────────────────
    test_data, vocab_size, eos_id, bytes_per_token = prepare_test_data(args, log)
    if vocab_size > cfg.vocab_size:
        # Tokenizer's real vocab > model's padded vocab → some token IDs
        # would index out-of-bounds in the embedding. Fail loud.
        raise RuntimeError(
            f"Tokenizer vocab {vocab_size} exceeds model vocab {cfg.vocab_size}. "
            f"The checkpoint was trained on a different (smaller) tokenizer."
        )

    test_ds = PackedTokenDataset(test_data)
    global_batch = args.batch_size * num_devices
    test_loader = DataLoader(
        test_ds, batch_size=global_batch,
        shuffle=False, drop_last=True,
        num_workers=args.num_workers,
        pin_memory=not HAS_XLA,
        persistent_workers=args.num_workers > 0,
    )
    if HAS_XLA:
        test_iter_loader = pl.MpDeviceLoader(
            test_loader, device, device_prefetch_size=args.prefetch,
        )
    else:
        test_iter_loader = test_loader
    log(f"Test set: {len(test_ds):,} chunks × {test_data.shape[1]} tokens, "
        f"global_batch={global_batch}, n_batches={len(test_loader)}")

    autocast_ctx = (
        torch.autocast(device_type=("xla" if HAS_XLA else device.type),
                       dtype=torch.bfloat16)
        if args.bf16 else _nullcontext()
    )

    # ── 1. Quality metrics on the test set ─────────────────────────────────
    log("\n══════ TEST SET EVALUATION ══════")
    test_metrics = evaluate_test(
        model, test_iter_loader, cfg.vocab_size, device,
        bytes_per_token=bytes_per_token, autocast_ctx=autocast_ctx,
        spmd_mesh=spmd_mesh, num_devices=num_devices, log=log,
    )
    log(f"  loss      = {test_metrics['test/loss']:.4f}  nats/token")
    log(f"  perplexity= {test_metrics['test/perplexity']:.4f}")
    log(f"  bpb       = {test_metrics['test/loss_bits_per_byte']:.4f}  "
        f"(bytes/token={bytes_per_token:.3f})")
    log(f"  top-1     = {test_metrics['test/top1_accuracy']*100:.2f}%")
    log(f"  top-5     = {test_metrics['test/top5_accuracy']*100:.2f}%")
    log(f"  samples   = {test_metrics['test/n_samples']:,}  "
        f"tokens={test_metrics['test/n_tokens']:,}  "
        f"elapsed={test_metrics['test/elapsed_seconds']:.1f}s  "
        f"({test_metrics['test/throughput_tps']:.0f} tok/s)")

    # ── 2. Inference benchmark (decode/prefill throughput, state size) ─────
    inf_metrics: dict = {}
    if not args.no_inference_bench and is_master:
        log("\n══════ INFERENCE BENCHMARK ══════")
        try:
            # Drop prefill lengths exceeding the trained seq_len.
            prefill_lens = tuple(L for L in args.inference_prefill_lens
                                 if L <= cfg.seq_len)
            inf_metrics = _benchmark_inference(
                model, cfg, device,
                prefill_lens=prefill_lens or (cfg.seq_len,),
                decode_tokens=args.inference_decode_tokens,
                batch_size=1,
            )
            for k in sorted(inf_metrics):
                v = inf_metrics[k]
                if isinstance(v, float):
                    log(f"  {k:42s} = {v:,.4g}")
                else:
                    log(f"  {k:42s} = {v}")
        except Exception as e:
            log(f"inference benchmark failed: {e!r}")

    # ── 3. Sample generations ──────────────────────────────────────────────
    samples: list[tuple[str, str]] = []
    if not args.no_samples and is_master and HAS_TRANSFORMERS:
        log("\n══════ SAMPLE GENERATIONS ══════")
        try:
            sample_tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
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
                    log(f"  ✏  [{prompt!r}]")
                    log(f"     → {txt!r}")
                except Exception as e:
                    log(f"  ✗ sample for {prompt!r} failed: {e!r}")
        except Exception as e:
            log(f"failed to load sampling tokenizer: {e!r}")

    # ── 4. Push everything to wandb ────────────────────────────────────────
    if wandb_enabled:
        try:
            wandb.log({**test_metrics, **inf_metrics})
            # Also surface the headline numbers in the run summary so they
            # appear at the top of the wandb run page.
            for k, v in {**test_metrics, **inf_metrics}.items():
                wandb.run.summary[k] = v
            if samples:
                table = wandb.Table(columns=["prompt", "generation"])
                for prompt, txt in samples:
                    table.add_data(prompt, txt)
                wandb.log({"test/samples": table})
            log(f"\nResults logged to {wandb.run.url}")
        except Exception as e:
            log(f"wandb.log failed: {e!r}")
        finally:
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
