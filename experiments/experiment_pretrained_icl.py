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
                        "tokenizer_name": "TODO: tokenizer name or path"
                        },
        "transformer": {
            "model_id": "SimpleStories/SimpleStories-5M"
            },
    },
    {
        "label": "35M_SimpleStories",
        "ssm":         {
            "ckpt_path": "TODO: SSM ~130M trained on Pile"
            ,"tokenizer_name": "TODO: tokenizer name or path"
            },
        "transformer": {"model_id": "SimpleStories/SimpleStories-35M"},
    },
    {
        "label": "~130M_pile",
        "ssm":         {"model_id": "TODO: SSM ~130M trained on Pile"},
        "transformer": {"model_id": "TODO: Transformer ~130M trained on Pile"},
    },
    {
        "label": "~370M_pile",
        "ssm":         {"model_id": "TODO: SSM ~370M trained on Pile"},
        "transformer": {"model_id": "TODO: Transformer ~370M trained on Pile"},
    },
    {
        "label": "~790M_pile",
        "ssm":         {"model_id": "TODO: SSM ~790M trained on Pile"},
        "transformer": {"model_id": "TODO: Transformer ~790M trained on Pile"},
    },
    {
        "label": "~1.4B_pile",
        "ssm":         {"model_id": "TODO: SSM ~1.4B trained on Pile"},
        "transformer": {"model_id": "TODO: Transformer ~1.4B trained on Pile"},
    },
    {
        "label": "~2.8B_pile",
        "ssm":         {"model_id": "TODO: SSM ~2.8B trained on Pile"},
        "transformer": {"model_id": "TODO: Transformer ~2.8B trained on Pile"},
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
