"""
Adapted from https://github.com/roeehendel/icl_task_vectors
"""
# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import time
import json
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from task_evaluation import calculate_accuracy_on_datasets

import random
from typing import Any, List, Optional, Iterable

from fewshot_data import FewShotDataset
from transformers import PreTrainedTokenizer

import torch
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class LinguisticTask():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mapping_type: str,
        mapping_name: str,
        allow_prefix: bool = False,
    ):
        super().__init__(tokenizer)
        self.mapping_type = mapping_type
        self.mapping_name = mapping_name
        self.allow_prefix = allow_prefix

        mapping_file = os.path.join(config.DATA_DIR, mapping_type, f"{mapping_name}.json")
        with open(mapping_file) as f:
            mapping = json.load(f)

        if allow_prefix:
            self.mapping = mapping
        else:
            num_before_filter = len(mapping)

            mapping_leading_space = {f" {k}": f" {v}" for k, v in mapping.items()}

            filtered_mapping = filter_single_token_outputs(tokenizer, mapping)
            filtered_mapping_leading_space = filter_single_token_outputs(tokenizer, mapping_leading_space)

            if len(filtered_mapping_leading_space) >= 0.7 * len(filtered_mapping):
                self.mapping = filtered_mapping_leading_space
            else:
                self.mapping = filtered_mapping

            if len(self.mapping) < MIN_NUM_EXAMPLES:
                print(
                    f"WARNING: mapping {mapping_name} has only {len(self.mapping)} examples after filtering "
                    f"({num_before_filter} before)"
                )

    def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
        input_space = list(self.mapping.keys())
        return random.sample(set(input_space) - set(exclude), num_inputs)

    def calc_output(self, inp) -> str:
        return self.mapping[inp]

    def num_examples(self) -> int:
        return len(self.mapping)


    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        output1, output2 = output1.strip(), output2.strip()

        if self.allow_prefix:
            nonempy = len(output1) > 0 and len(output2) > 0
            return nonempy and (output1.startswith(output2) or output2.startswith(output1))
        return output1 == output2

    def calc_test_output(self, inp: Any) -> Any:
        return self.calc_output(inp)

    def create_datasets(self, num_datasets: int, num_examples: int) -> List[FewShotDataset]:
        return [self.create_dataset(num_examples) for _ in range(num_datasets)]

    def create_dataset(self, num_examples: int, test_input: Optional[Any] = None) -> FewShotDataset:
        if test_input is None:
            test_input = self.sample_inputs(1)[0]
        test_output = self.calc_test_output(test_input)

        train_inputs = self.sample_inputs(num_examples, exclude=[test_input])
        train_outputs = [self.calc_output(x) for x in train_inputs]

        train_inputs = [str(x) for x in train_inputs]
        train_outputs = [str(x) for x in train_outputs]
        test_input = str(test_input)
        test_output = str(test_output)

        return FewShotDataset(
            train_inputs,
            train_outputs,
            test_input,
            test_output,
        )

LINGUISTIC_TASKS = [
"linguistic_present_simple_gerund": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_gerund"},
    },
    "linguistic_present_simple_past_simple": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_simple"},
    },
    "linguistic_present_simple_past_perfect": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_perfect"},
    },
    "linguistic_singular_plural": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "singular_plural"},
    },
    "linguistic_plural_singular": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "plural_singular"},
    },
    "linguistic_antonyms": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "antonyms"},
    }
]

def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def get_task_by_name(tokenizer: PreTrainedTokenizer, task_name: str) -> LinguisticTask:
    task_args = LINGUISTIC_TASKS[task_name]
    task = LinguisticTask(**task_args["task_kwargs"], tokenizer=tokenizer)
    return task


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

    return predictions

def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL 

    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    icl_predictions = run_icl(model, tokenizer, task, test_datasets)
    
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
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Evaluating model:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
        print(f"Dev Accuracy by layer: ", end="")
        for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}, ", end="")
        print()
        print("Time:", time.time() - tic)

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": num_examples,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)

