import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from mamba_inference import load_hf_model, load_ckpt_model

from icl_tasks import run_main_experiment

# Each entry supports either:
#   model_id: str          — downloads from HuggingFace
#   ckpt_path + tokenizer_name: str — loads a local .pt checkpoint

 
EXPERIMENTS = [
    {
        "label": "5M_SimpleStories",
        "ssm":         {
                        "ckpt_path": "experiments/checkpoints/mamba_simplestories_5m_final.pt",
                        "tokenizer_name": "SimpleStories/SimpleStories-5M"
                        },
        "transformer": {
            "model_id": "SimpleStories/SimpleStories-5M"
            },
    },
    {
        "label": "35M_SimpleStories",
        "ssm":         {
            "ckpt_path": "TODO: SSM ~130M trained on Pile"
            ,"tokenizer_name": "SimpleStories/SimpleStories-5M"
            },
        "transformer": {"model_id": "SimpleStories/SimpleStories-35M"},
    },

# Mamba (state-spaces) vs Pythia (EleutherAI) — both trained on the Pile with 300B tokens.
# The Mamba paper explicitly used Pythia as the transformer baseline, so these are
# as apples-to-apples as it gets for SSM vs transformer comparisons.
    {
        "label": "~130M_pile",
        "ssm":         {"model_id": "state-spaces/mamba-130m-hf"},   # 130M
        "transformer": {"model_id": "EleutherAI/pythia-160m"},        # 160M
    },
    {
        "label": "~370M_pile",
        "ssm":         {"model_id": "state-spaces/mamba-370m-hf"},    # 370M
        "transformer": {"model_id": "EleutherAI/pythia-410m"},        # 410M
    },
    {
        "label": "~790M_pile",
        "ssm":         {"model_id": "state-spaces/mamba-790m-hf"},    # 790M
        "transformer": {"model_id": "EleutherAI/pythia-1b"},          # 1B
    },
    {
        "label": "~1.4B_pile",
        "ssm":         {"model_id": "state-spaces/mamba-1.4b-hf"},    # 1.4B
        "transformer": {"model_id": "EleutherAI/pythia-1.4b"},        # 1.4B (exact match)
    },
    {
        "label": "~2.8B_pile",
        "ssm":         {"model_id": "state-spaces/mamba-2.8b-hf"},    # 2.8B
        "transformer": {"model_id": "EleutherAI/pythia-2.8b"},        # 2.8B (exact match)
    },
]


def run_experiments(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    for exp in EXPERIMENTS:
        label = exp["label"]
        print(f"\n{'='*60}")
        print(f"Experiment: {label}")

        for arch, cfg in [("ssm", exp["ssm"]), ("transformer", exp["transformer"])]:
            if "ckpt_path" in cfg:
                source = cfg["ckpt_path"]
                if source.startswith("TODO"):
                    print(f"  Skipping {arch} ({label}): placeholder not set")
                    continue
                print(f"\n  Loading {arch} from checkpoint: {source}")
                model, tokenizer = load_ckpt_model(
                    source, cfg["tokenizer_name"], device=device
                )
            else:
                source = cfg["model_id"]
                if source.startswith("TODO"):
                    print(f"  Skipping {arch} ({label}): placeholder not set")
                    continue
                print(f"\n  Loading {arch} from HuggingFace: {source}")
                model, tokenizer = load_hf_model(source, device=device)


            run_main_experiment(
                model,
                tokenizer,
                model_type=arch,
                model_variant=label,
            )

            # Free memory between runs
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run_experiments()
