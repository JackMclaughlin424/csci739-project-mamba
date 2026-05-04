from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer

from fewshot_data import FewShotDataset, FewShotFormat


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
    return tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False)


def batch_generate(
    model,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    generate_kwargs: Optional[Dict] = None,
    batch_size: int = 8,
) -> torch.Tensor:
    generate_kwargs = dict(generate_kwargs or {})
    max_new_tokens = generate_kwargs.pop("max_new_tokens", 1)

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"]

    all_new_ids = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            B, L = batch_ids.shape

            # Single parallel forward over the full prompt; take the last position's logits.
            logits = model(batch_ids)           # (B, L, V)
            next_token = logits[:, -1, :].argmax(dim=-1)   # (B,)

            if max_new_tokens == 1:
                all_new_ids.append(next_token.unsqueeze(1).cpu())
                continue

            # For multiple new tokens, populate the recurrent cache then step.
            caches = model.allocate_inference_cache(
                batch_size=B, dtype=torch.float32, device=device,
            )
            for t in range(L):
                _, caches = model.step(batch_ids[:, t], caches)

            generated = [next_token.unsqueeze(1)]
            cur_token = next_token
            for _ in range(max_new_tokens - 1):
                logits, caches = model.step(cur_token, caches)
                cur_token = logits.argmax(dim=-1)
                generated.append(cur_token.unsqueeze(1))

            all_new_ids.append(torch.cat(generated, dim=1).cpu())

    return torch.cat(all_new_ids, dim=0)


def decode_predictions(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    few_shot_format: FewShotFormat = None,
) -> List[str]:
    few_shot_format = few_shot_format or FewShotFormat()
    new_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    answers = [tok.split(few_shot_format.example_separator)[0] for tok in new_tokens]
    return answers
