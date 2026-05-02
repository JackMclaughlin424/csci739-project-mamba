"""
Helper functions for running experiments and scoring generative metrics

metrics:
- perplexity
- ROUGE-L
- BLEU
-BERTScore

experiments:
- ICL
    - present to gerund
    - present to past
    - singular to plural
    - antonyms
"""

import torch
import torch.nn.functional as F

from rouge_score import rouge_scorer


def compute_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    """
    Average perplexity = exp(mean cross-entropy loss) over all non-padding tokens.
    Pads batches to the longest sequence in each chunk; masks padding from the loss.
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids).logits  # (B, L, V)

            # Shift by one position for next-token prediction targets
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            total_loss += (loss * shift_mask.view(-1)).sum().item()
            total_tokens += shift_mask.sum().item()

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def batch_rouge_l(predictions, references):
    """Mean ROUGE-L F1 over paired prediction/reference lists."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0
