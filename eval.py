"""
eval.py — Evaluate a Mamba checkpoint and optionally compare against a baseline.

Usage:
    # Mamba only
    python eval.py --checkpoint mamba_simplestories_5m.pt

    # Side-by-side with the matched transformer
    python eval.py --checkpoint mamba_simplestories_5m.pt \
                   --baseline SimpleStories/SimpleStories-5M
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mamba.mamba_llm_tpu import MambaLMConfig, MambaLMHeadModel


# ── Model loading ─────────────────────────────────────────────────────────────

def load_mamba(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg     = MambaLMConfig(**payload["config"])
    model   = MambaLMHeadModel(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    meta = {k: payload.get(k) for k in ("global_step", "epoch", "best_val", "best_val_ppl")}
    return model, cfg, meta


def load_transformer(name, device):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    return model


# ── Data ──────────────────────────────────────────────────────────────────────

def build_test_data(dataset, split, tokenizer, seq_len, max_stories, lowercase):
    ds = load_dataset(dataset, split=split)
    text_col = next((c for c in ("story", "text") if c in ds.column_names), ds.column_names[0])
    if max_stories:
        ds = ds.select(range(min(max_stories, len(ds))))

    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    stream = []
    for ex in tqdm(ds, desc="tokenizing", leave=False):
        text = ex[text_col].lower() if lowercase else ex[text_col]
        stream.extend(tokenizer.encode(text, add_special_tokens=False) + [eos])

    chunk = seq_len + 1
    n     = len(stream) // chunk
    data  = torch.tensor(stream[: n * chunk], dtype=torch.long).view(n, chunk)
    print(f"  {len(ds):,} stories → {n:,} chunks × {seq_len} tokens")
    return data


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, batch_size, device):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    losses, top1s, top5s = [], [], []

    for batch in tqdm(loader, desc="  eval", leave=False):
        x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)

        out    = model(x)
        logits = out.logits if hasattr(out, "logits") else out  # Mamba vs HF

        flat_l = logits.reshape(-1, logits.size(-1))
        flat_y = y.reshape(-1)

        losses.append(F.cross_entropy(flat_l, flat_y).item())
        top1s.append((flat_l.argmax(-1) == flat_y).float().mean().item())
        top5s.append((flat_l.topk(5, -1).indices == flat_y.unsqueeze(-1)).any(-1).float().mean().item())

    avg_loss = sum(losses) / len(losses)
    return {
        "loss":        avg_loss,
        "perplexity":  math.exp(min(avg_loss, 20)),
        "top1":        sum(top1s) / len(top1s),
        "top5":        sum(top5s) / len(top5s),
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_result(label, metrics, meta=None):
    print(f"\n{'═' * 52}")
    print(f"  {label}")
    if meta and meta.get("global_step"):
        v = meta.get("best_val")
        print(f"  step {meta['global_step']:,}  epoch {meta.get('epoch', '?')}  "
              f"best_val {v:.4f}" if v else "")
    print(f"{'─' * 52}")
    print(f"  Loss (nats)   {metrics['loss']:>10.4f}")
    print(f"  Perplexity    {metrics['perplexity']:>10.2f}")
    print(f"  Top-1 acc     {metrics['top1']:>10.4f}  ({metrics['top1']*100:.1f}%)")
    print(f"  Top-5 acc     {metrics['top5']:>10.4f}  ({metrics['top5']*100:.1f}%)")
    print(f"{'═' * 52}")


def print_comparison(mamba_m, xfmr_m, xfmr_label):
    print(f"\n{'═' * 52}")
    print(f"  {'COMPARISON'}")
    print(f"  {'Metric':<20} {'Mamba':>10} {xfmr_label:>12}")
    print(f"  {'─' * 46}")
    for k, label in [("loss", "Loss"), ("perplexity", "Perplexity"),
                     ("top1", "Top-1 acc"), ("top5", "Top-5 acc")]:
        print(f"  {label:<20} {mamba_m[k]:>10.4f} {xfmr_m[k]:>12.4f}")
    print(f"{'═' * 52}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,  help="Path to .pt checkpoint from tpu_train.py")
    p.add_argument("--baseline",     default=None,   help="HuggingFace model name to compare against")
    p.add_argument("--tokenizer",    default="SimpleStories/SimpleStories-5M")
    p.add_argument("--dataset",      default="lennart-finke/SimpleStories")
    p.add_argument("--split",        default="test")
    p.add_argument("--seq_len",      type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--max_stories",  type=int, default=2000, help="0 = full split")
    p.add_argument("--no_lowercase", action="store_true")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device    = torch.device(args.device)
    lowercase = not args.no_lowercase
    print(f"Device: {device}")

    # Tokenizer + data
    print(f"\nTokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Dataset:   {args.dataset}  split={args.split}")
    data = build_test_data(args.dataset, args.split, tokenizer,
                           args.seq_len, args.max_stories or None, lowercase)

    # Mamba
    print(f"\nLoading Mamba: {args.checkpoint}")
    mamba, cfg, meta = load_mamba(args.checkpoint, device)
    n_params = sum(p.numel() for p in mamba.parameters())
    print(f"  n_layer={cfg.n_layer}  d_input={cfg.d_input}  d_model={cfg.d_model}")
    print(f"  params: {n_params:,}")
    mamba_metrics = evaluate(mamba, data, args.batch_size, device)
    print_result(f"Mamba  ({os.path.basename(args.checkpoint)})", mamba_metrics, meta)

    # Optional baseline
    if args.baseline:
        print(f"\nLoading baseline: {args.baseline}")
        transformer   = load_transformer(args.baseline, device)
        n_xfmr        = sum(p.numel() for p in transformer.parameters())
        print(f"  params: {n_xfmr:,}")
        xfmr_metrics  = evaluate(transformer, data, args.batch_size, device)
        print_result(args.baseline, xfmr_metrics)
        print_comparison(mamba_metrics, xfmr_metrics, "Transformer")


if __name__ == "__main__":
    main()
