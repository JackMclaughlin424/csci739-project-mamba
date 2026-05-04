from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer

from icl_task_vectors.core.data.datasets.few_shot_dataset import FewShotDataset
from icl_task_vectors.core.data.datasets.few_shot_format import FewShotFormat



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
                    logits = model(real_ids)
                    

                    next_token = logits[0, -1, :].argmax()
                    all_new_ids.append(next_token.view(1, 1))
            else:
                logits = model(batch_ids)
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

