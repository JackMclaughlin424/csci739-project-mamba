from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer

from icl_task_vectors.core.data.datasets.few_shot_dataset import FewShotDataset
from icl_task_vectors.core.data.datasets.few_shot_format import FewShotFormat


def _extract_logits(output) -> torch.Tensor:
    # HuggingFace CausalLM models return a ModelOutput dataclass; Mamba returns a raw tensor
    if isinstance(output, torch.Tensor):
        return output
    return output.logits



def tokenize_datasets(
    tokenizer: PreTrainedTokenizer,
    datasets: List[FewShotDataset],
    few_shot_format: FewShotFormat = None,
    format_dataset_kwargs: Optional[dict] = None,
) -> Dict:
    few_shot_format = few_shot_format or FewShotFormat()
    format_dataset_kwargs = format_dataset_kwargs or {}
    prompts = few_shot_format.format_datasets(datasets, **format_dataset_kwargs)
    # Left-pad so position -1 always corresponds to the last real token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, add_special_tokens=False)




def batch_generate(
    model,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    generate_kwargs: Optional[Dict] = None,
    batch_size: int = 8,
) -> torch.Tensor:
    try:
        import torch_xla.core.xla_model as xm
        HAS_XLA = True
    except ImportError:
        HAS_XLA = False

    generate_kwargs = dict(generate_kwargs or {})
    max_new_tokens = generate_kwargs.pop("max_new_tokens", 1)

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"]

    all_new_ids = []
    attention_mask = inputs.get("attention_mask")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            if attention_mask is not None:
                batch_mask = attention_mask[i : i + batch_size].to(device)
                for j in range(batch_ids.shape[0]):
                    real_ids = batch_ids[j][batch_mask[j].bool()].unsqueeze(0)
                    # debug: inspect what tokens the model actually receives
                    # if j == 0:
                    #     print("Input token IDs:", real_ids[0].tolist())
                    #     print("Decoded input:", tokenizer.decode(real_ids[0].tolist(), skip_special_tokens=False))
                    raw_output = model(real_ids)
                    logits = _extract_logits(raw_output)
                    next_token = logits[0, -1, :].argmax()
                    all_new_ids.append(next_token.view(1, 1))
            else:
                raw_output = model(batch_ids)
                logits = _extract_logits(raw_output)
                next_token = logits[:, -1, :].argmax(dim=-1)
                all_new_ids.append(next_token.unsqueeze(1))


            if HAS_XLA:
                xm.mark_step()


    # Single transfer off-device after all batches are done.
    return torch.cat(all_new_ids, dim=0).cpu()



def decode_predictions(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    few_shot_format: FewShotFormat = None,
) -> List[str]:
    few_shot_format = few_shot_format or FewShotFormat()
    new_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    # print("Raw predicted tokens:", new_tokens[:5])
    answers = [tok.split(few_shot_format.example_separator)[0] for tok in new_tokens]
    return answers


def load_hf_model(model_name: str, device: str = "cpu"):
    # Loads any causal LM from HuggingFace hub; compatible with the existing batch_generate pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.tie_weights()
    model.to(device)

    model.eval()
    return model, tokenizer


def load_ckpt_model(ckpt_path: str, tokenizer_name: str, device: str = "cpu"):
    # Loads a custom MambaLMHeadModel from a .pt checkpoint saved by tpu_train.py
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mamba.mamba_llm_tpu import MambaLMHeadModel, MambaLMConfig
    from transformers import AutoTokenizer

    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = MambaLMConfig(**payload["config"])
    model = MambaLMHeadModel(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer
