"""
Adapted from https://github.com/roeehendel/icl_task_vectors
"""
# This must be first
# from dotenv import load_dotenv


# load_dotenv(".env")

import sys
import os
import pickle
import time
import json
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer


from icl_task_vectors.core.analysis.evaluation import calculate_accuracy_on_datasets
from icl_task_vectors.core.data.datasets.few_shot_dataset import FewShotDataset

from icl_task_vectors.core.data.task_helpers import ALL_TASKS, get_all_tasks, get_task_by_name

import random
from typing import Any, List, Optional, Iterable

from transformers import PreTrainedTokenizer

import torch
import numpy as np

# our imports
from mamba_inference import batch_generate, decode_predictions,  tokenize_datasets


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

MIN_NUM_EXAMPLES = 70

DATA_DIR = "linguistic_mappings"

LINGUISTIC_TASKS = {
    "linguistic_present_simple_gerund": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_gerund"}
    },
    "linguistic_present_simple_past_simple": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_simple"}
    },
    "linguistic_present_simple_past_perfect": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_perfect"}
    },
    "linguistic_singular_plural": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "singular_plural"}
    },
    "linguistic_plural_singular": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "plural_singular"}
    },
    "linguistic_antonyms": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "antonyms"}
    }
}



# def get_all_tasks(tokenizer: PreTrainedTokenizer):
#     tasks = {task_name: get_task_by_name(tokenizer, task_name) for task_name in LINGUISTIC_TASKS}
#     return tasks



def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_datasets: List[FewShotDataset],
    include_train: bool = True,
) -> List[str]:
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs=format_dataset_kwargs)
    new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": 1})
    predictions = decode_predictions(new_ids, tokenizer)
    print("Sample predictions:", predictions[:5])
    print("Sample expected:", [d.test_output for d in test_datasets[:5]])
    return predictions


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL 

    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    icl_predictions = run_icl(model, tokenizer, test_datasets)
    
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    
    # dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    # tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
    #     model,
    #     tokenizer,
    #     task,
    #     test_datasets,
    #     dev_datasets,
    # )
    # accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    # accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)

    # tv_ordered_tokens_by_layer = {}
    # try:
    #     for layer_num in tv_dev_accuracy_by_layer.keys():
    #         task_hidden = task_hiddens.mean(axis=0)[layer_num]
    #         logits = hidden_to_logits(model, task_hidden)
    #         tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    # except Exception as e:
    #     print("Error:", e)

    return accuracies       #, tv_ordered_tokens_by_layer


def run_main_experiment(
    model,
    tokenizer,
    model_type = "mamba", model_variant = "5m"
) -> None:
    print(f"Evaluating {model_type}_{model_variant} on ICL...")

    results_file = f"experiments/results/{model_type}_{model_variant}_results.pkl"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # if os.path.exists(results_file):
    #     with open(results_file, "rb") as f:
    #         results = pickle.load(f)
    # else:
    results = {}

    # limit_gpus(range(0, 8))


    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5

    for i, task_name in enumerate(ALL_TASKS):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies = evaluate_task(model, tokenizer, task_name, num_examples)

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        print("Time:", time.time() - tic)

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": num_examples,
            "icl_accuracy": accuracies["icl"],
            # "tv_accuracy": accuracies["tv"],
            # "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            # "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def main():
    import argparse
    from transformers import AutoTokenizer

    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from mamba.mamba_llm_tpu import MambaLMHeadModel, MambaLMConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .pt checkpoint saved by tpu_train.py")
    parser.add_argument("--model_variant", default="5m", help="Label used in the results filename")
    parser.add_argument("--tokenizer_name", default="SimpleStories/SimpleStories-5M")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print("Device: " + args.device)
    payload = torch.load(args.model_path, map_location="cpu")
    cfg = MambaLMConfig(**payload["config"])
    model = MambaLMHeadModel(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    run_main_experiment(model, tokenizer, model_type="mamba", model_variant=args.model_variant)



if __name__=="__main__":
    main()