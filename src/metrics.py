import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import spearmanr
import torch
from src.attribution import find_token_span


def compute_has(
    scores: np.ndarray,
    hint_span: Tuple[int, int],
    prompt_span: Tuple[int, int],
) -> float:
    # Returns a value between 0 (hint completely ignored) 
    # and 1 (hint took 100% of prompt attribution)

    # HAS = sum_{i in Hint} |Attr(i)| / sum_{j in Prompt} |Attr(j)|
    abs_scores = np.abs(scores)
    hint_start, hint_end = hint_span
    prompt_start, prompt_end = prompt_span

    if prompt_end <= prompt_start or len(abs_scores) == 0:
        return 0.0

    prompt_scores = abs_scores[prompt_start:min(prompt_end, len(abs_scores))]
    total_prompt_attr = float(np.sum(prompt_scores))
    if total_prompt_attr == 0.0 or np.isnan(total_prompt_attr):
        return 0.0

    hint_scores = abs_scores[hint_start:min(hint_end, len(abs_scores))]
    hint_attr = float(np.sum(hint_scores))

    has_val = hint_attr / total_prompt_attr
    return float(np.clip(has_val, 0.0, 1.0))


def compute_rank_stability(
    clean_scores: np.ndarray,
    biased_scores: np.ndarray,
    clean_q_span: Tuple[int, int],
    biased_q_span: Tuple[int, int],
) -> float:

    # Isolates the attribution scores 
    # belonging strictly to the original question words in the clean vs. biased prompt

    # Computes Spearman's rank correlation, 
    # tells us whether the injected bias corrupted the 
    # model's internal feature importance on the main problem
    
    cq_start, cq_end = clean_q_span
    bq_start, bq_end = biased_q_span

    clean_vec = np.abs(clean_scores[cq_start:cq_end])
    biased_vec = np.abs(biased_scores[bq_start:bq_end])

    min_len = min(len(clean_vec), len(biased_vec))
    if min_len < 2:
        return 0.0

    clean_aligned = clean_vec[:min_len]
    biased_aligned = biased_vec[:min_len]

    if np.all(clean_aligned == clean_aligned[0]) or np.all(biased_aligned == biased_aligned[0]):
        return 0.0

    corr, _ = spearmanr(clean_aligned, biased_aligned)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_comprehensiveness(
    model: Any,
    tokenizer: Any,
    input_text: str,
    scores: np.ndarray,
    prompt_span: Tuple[int, int],
    target_token_id: int,
    top_pct: float = 0.10,
    device: str = "cuda",
) -> float:

    # Erasure test
    # Identifies the top 10% most attributed tokens according to that method and masks them with
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    # Baseline probability of target token
    with torch.no_grad():
        orig_logits = model(input_ids).logits[0, -1]
        p_orig = float(torch.softmax(orig_logits, dim=-1)[target_token_id].item())

    p_start, p_end = prompt_span
    p_end = min(p_end, seq_len)
    prompt_scores = np.abs(scores[p_start:p_end])

    num_to_mask = max(1, int(len(prompt_scores) * top_pct))
    top_indices_in_prompt = np.argsort(prompt_scores)[-num_to_mask:]
    tokens_to_mask = [p_start + idx for idx in top_indices_in_prompt]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    masked_ids = input_ids.clone()
    for tidx in tokens_to_mask:
        masked_ids[0, tidx] = pad_id

    # Masked probability of target token
    with torch.no_grad():
        masked_logits = model(masked_ids).logits[0, -1]
        p_masked = float(torch.softmax(masked_logits, dim=-1)[target_token_id].item())

    # Comprehensiveness is the drop in target confidence
    drop = p_orig - p_masked
    return float(drop)
