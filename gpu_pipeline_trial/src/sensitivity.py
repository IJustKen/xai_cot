import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from src.attribution import get_embedding_layer, attribute_integrated_gradients, find_token_span
from src.metrics import compute_has


def evaluate_ig_sensitivity(
    model: Any,
    tokenizer: Any,
    input_text: str,
    target_token_id: int,
    hint_span: Tuple[int, int],
    prompt_span: Tuple[int, int],
    steps_list: List[int] = [10, 25, 50, 100],
    baselines: List[str] = ["pad", "zero"],
    device: str = "cuda",
) -> List[Dict[str, Any]]:

    # Tests IG across varying Riemann approximation steps
    # AI helped with some of this as it did with IG
    
    results = []

    # First get the reference score at maximum steps (pad baseline, max steps)
    max_steps = max(steps_list)
    ref_run = attribute_integrated_gradients(
        model, tokenizer, input_text, target_token_id=target_token_id, m_steps=max_steps, baseline_type="pad", device=device
    )
    ref_scores = ref_run["scores"]

    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embed_layer = get_embedding_layer(model)
    input_embeds = embed_layer(input_ids).detach()

    with torch.no_grad():
        fx = model(inputs_embeds=input_embeds).logits[0, -1, target_token_id].item()

    for base_type in baselines:
        if base_type == "pad":
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_ids = torch.full_like(input_ids, pad_id)
            base_embeds = embed_layer(pad_ids).detach()
        else:
            base_embeds = torch.zeros_like(input_embeds)

        with torch.no_grad():
            fx0 = model(inputs_embeds=base_embeds).logits[0, -1, target_token_id].item()

        theoretical_delta = fx - fx0

        for m in steps_list:
            run = attribute_integrated_gradients(
                model,
                tokenizer,
                input_text,
                target_token_id=target_token_id,
                m_steps=m,
                baseline_type=base_type,
                device=device,
            )
            scores = run["scores"]

            # Completeness approximation error: |sum(attr) - (F(x) - F(x0))|
            sum_attr = float(np.sum(scores))
            approx_error = float(abs(sum_attr - theoretical_delta))

            # Stability: rank correlation with reference score at max steps
            corr, _ = spearmanr(scores, ref_scores)
            stability = float(corr) if not np.isnan(corr) else 1.0

            has_val = compute_has(scores, hint_span, prompt_span)

            results.append({
                "m_steps": m,
                "baseline_type": base_type,
                "approx_error": approx_error,
                "stability_spearman": stability,
                "has": has_val,
                "theoretical_delta": theoretical_delta,
                "sum_attr": sum_attr,
            })

    return results


def attribute_reasoning_sequence(
    model: Any,
    tokenizer: Any,
    full_text: str,
    prompt_len: int,
    m_steps: int = 50,
    device: str = "cuda",
) -> Dict[str, Any]:

    # Computes attribution with respect to the 
    # cross-entropy loss over the generated reasoning tokens rather 
    # than just the final answer token
    
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    embed_layer = get_embedding_layer(model)
    input_embeds = embed_layer(input_ids).detach()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    baseline_embeds = embed_layer(torch.full_like(input_ids, pad_id)).detach()
    delta = input_embeds - baseline_embeds

    # Intermediate reasoning token targets
    reasoning_ids = input_ids[0, prompt_len:-1]
    if len(reasoning_ids) == 0:
        return {
            "tokens": tokens,
            "token_ids": input_ids[0].tolist(),
            "scores": np.zeros(seq_len),
            "method": "reasoning_aggregated_ig",
        }

    accumulated_grads = torch.zeros_like(input_embeds)
    alphas = torch.linspace(1.0 / m_steps, 1.0, m_steps, device=device)

    for alpha in alphas:
        step_embeds = baseline_embeds + alpha * delta
        step_embeds = step_embeds.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=step_embeds)
        logits = outputs.logits[0, prompt_len - 1 : -2]  # logits predicting reasoning tokens

        # Target is cumulative negative log-likelihood of reasoning tokens
        loss = F.cross_entropy(logits, reasoning_ids)

        model.zero_grad()
        loss.backward()
        accumulated_grads += step_embeds.grad.detach()

    avg_grads = accumulated_grads / m_steps
    # Higher attribution means higher impact on generating reasoning tokens
    scores = (delta[0] * avg_grads[0]).sum(dim=-1).detach().cpu().numpy()

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "scores": scores,
        "method": "reasoning_aggregated_ig",
    }


def run_target_conditioning_ablation(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    target_token_id: int,
    hint_span: Tuple[int, int],
    prompt_span: Tuple[int, int],
    m_steps: int = 50,
    device: str = "cuda",
) -> Dict[str, Any]:

    # Compares HAS when attributing the final answer token
    #  vs. 
    # attributing the aggregated reasoning sequence


    # if reasoning tokens absorb the gradient, 
    # HAS will be way higher when attributing 
    # the reasoning target than the answer token target.
    
    full_text = prompt_text + " " + response_text
    prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    # Target 1 - Strictly final answer token
    ans_run = attribute_integrated_gradients(
        model, tokenizer, full_text, target_token_id=target_token_id, m_steps=m_steps, device=device
    )
    ans_has = compute_has(ans_run["scores"], hint_span, prompt_span)

    # Target 2 - Aggregated reasoning sequence
    reas_run = attribute_reasoning_sequence(
        model, tokenizer, full_text, prompt_len=prompt_len, m_steps=m_steps, device=device
    )
    reas_has = compute_has(reas_run["scores"], hint_span, prompt_span)

    return {
        "final_answer_has": ans_has,
        "reasoning_aggregated_has": reas_has,
        "dilution_delta": ans_has - reas_has,
    }
