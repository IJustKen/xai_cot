import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


def find_token_span(tokenizer: Any, full_text: str, substring: str) -> Tuple[int, int]:

    # Maps any text snippet (such as the hint sentence or original question) 
    # to its exact [start_token_idx, end_token_idx] in the tokenized sequence.

    # AI was used to assist here since we were not very familiar with tokenizer

    if not substring or substring not in full_text:
        return (0, 0)

    char_start = full_text.index(substring)
    char_end = char_start + len(substring)

    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding.get("offset_mapping", None)

    # In case offset mapping is not provided by fast tokenizer
    if offsets is None:
        prefix_ids = tokenizer(full_text[:char_start], add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(full_text[:char_end], add_special_tokens=True)["input_ids"]
        start_idx = max(0, len(prefix_ids) - 1)
        end_idx = max(start_idx + 1, len(target_ids))
        return (start_idx, end_idx)

    token_start = -1
    token_end = -1
    for idx, (start, end) in enumerate(offsets):
        if end > char_start and token_start == -1:
            token_start = idx
        if end >= char_end and token_end == -1:
            token_end = idx + 1
            break

    if token_start == -1:
        token_start = 0
    if token_end == -1:
        token_end = len(offsets)

    return (token_start, token_end)


def get_embedding_layer(model: Any) -> torch.nn.Module:

    # Grabs the input token embedding layer from Llama-3.2.

    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    
    raise AttributeError("Could not locate embedding layer in model.")


def attribute_raw_attention(
    model: Any,
    tokenizer: Any,
    input_text: str,
    target_token_id: Optional[int] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:

    # Performs a forward pass with output_attentions=True
    # Averages attention heads in the final layer and extracts 
    # attention from the final decision token back to all prompt tokens.
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    # Last layer attentions: shape [batch, num_heads, seq_len, seq_len]

    # print(outputs.shape)
    last_layer_attn = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
    avg_heads = last_layer_attn.mean(dim=0)      # [seq_len, seq_len]

    # Attention from the last token (target decision token) to all preceding tokens
    target_idx = -1
    attn_scores = avg_heads[target_idx].detach().cpu().numpy()

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "scores": attn_scores,
        "method": "raw_attention",
    }


def attribute_attention_rollout(
    model: Any,
    tokenizer: Any,
    input_text: str,
    target_token_id: Optional[int] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:

    # Recursively computes A_l^ = 0.5*A_l + 0.5*I
    # and matrix-multiplies across all transformer layers 
    # to trace the flow of attention through the entire network
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    # Initialize rollout matrix as identity
    rollout = torch.eye(seq_len, device=device)

    for layer_attn in outputs.attentions:
        # Average over attention heads: [seq_len, seq_len]
        attn_avg = layer_attn[0].mean(dim=0)
        # Add identity and renormalize
        a_hat = 0.5 * attn_avg + 0.5 * torch.eye(seq_len, device=device)
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(a_hat, rollout)

    # Attention rollout from the final token to all tokens
    scores = rollout[-1].detach().cpu().numpy()

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "scores": scores,
        "method": "attention_rollout",
    }


def attribute_input_x_gradient(
    model: Any,
    tokenizer: Any,
    input_text: str,
    target_token_id: Optional[int] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:

    # Computes the gradient of the target token's 
    # logit with respect to the input embeddings:
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    embed_layer = get_embedding_layer(model)
    embeddings = embed_layer(input_ids)
    embeddings = embeddings.clone().detach().requires_grad_(True)

    outputs = model(inputs_embeds=embeddings)
    logits = outputs.logits[0, -1]  # logits for predicting the next/final token

    if target_token_id is None:
        target_token_id = int(logits.argmax().item())

    target_score = logits[target_token_id]
    model.zero_grad()
    target_score.backward()

    # Input x Gradient - (grad * input) summed across embedding dimensions
    grads = embeddings.grad[0]
    input_embeds = embeddings[0].detach()
    ixg = (grads * input_embeds).sum(dim=-1).detach().cpu().numpy()

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "scores": ixg,
        "method": "input_x_gradient",
    }


def attribute_integrated_gradients(
    model: Any,
    tokenizer: Any,
    input_text: str,
    target_token_id: Optional[int] = None,
    m_steps: int = 50,
    baseline_type: str = "pad",
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:
    
    # Used AI to help with this

    # Path-integral method
    # Linearly interpolates embeddings from a baseline 
    # to the actual input x in m steps
    # and accumulates gradients along the path
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    seq_len = input_ids.shape[1]

    embed_layer = get_embedding_layer(model)
    input_embeds = embed_layer(input_ids).detach()

    # Reference baseline - pad token embedding or all zeroes
    if baseline_type == "pad":
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_input_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = embed_layer(pad_input_ids).detach()
    else:
        baseline_embeds = torch.zeros_like(input_embeds)

    delta = input_embeds - baseline_embeds

    # Riemann summation approximation
    accumulated_grads = torch.zeros_like(input_embeds)
    alphas = torch.linspace(1.0 / m_steps, 1.0, m_steps, device=device)

    # First forward to identify target token if not specified
    if target_token_id is None:
        with torch.no_grad():
            clean_logits = model(inputs_embeds=input_embeds).logits[0, -1]
            target_token_id = int(clean_logits.argmax().item())

    for alpha in alphas:
        step_embeds = baseline_embeds + alpha * delta
        step_embeds = step_embeds.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=step_embeds)
        target_score = outputs.logits[0, -1, target_token_id]

        model.zero_grad()
        target_score.backward()
        accumulated_grads += step_embeds.grad.detach()

    avg_grads = accumulated_grads / m_steps
    ig_scores = (delta[0] * avg_grads[0]).sum(dim=-1).detach().cpu().numpy()

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "scores": ig_scores,
        "method": "integrated_gradients",
    }


def attribute_gradient_shap(
    model: Any,
    tokenizer: Any,
    input_text: str,
    target_token_id: Optional[int] = None,
    n_samples: int = 10,
    stdevs: float = 0.1,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:

    # Approximates Shapley values by adding Gaussian noise to interpolated baseline 
    # embeddings and averaging gradients over n samples
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    embed_layer = get_embedding_layer(model)
    input_embeds = embed_layer(input_ids).detach()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    baseline_embeds = embed_layer(torch.full_like(input_ids, pad_id)).detach()

    if target_token_id is None:
        with torch.no_grad():
            clean_logits = model(inputs_embeds=input_embeds).logits[0, -1]
            target_token_id = int(clean_logits.argmax().item())

    delta = input_embeds - baseline_embeds
    accumulated_grads = torch.zeros_like(input_embeds)

    for _ in range(n_samples):
        # Sample random interpolation factor and add Gaussian perturbation
        alpha = torch.rand(1, device=device).item()
        noise = torch.randn_like(input_embeds) * stdevs
        step_embeds = baseline_embeds + alpha * delta + noise
        step_embeds = step_embeds.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=step_embeds)
        target_score = outputs.logits[0, -1, target_token_id]

        model.zero_grad()
        target_score.backward()
        accumulated_grads += step_embeds.grad.detach()

    avg_grads = accumulated_grads / n_samples
    shap_scores = (delta[0] * avg_grads[0]).sum(dim=-1).detach().cpu().numpy()

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "scores": shap_scores,
        "method": "gradient_shap",
    }


def attribute_direct_answer(
    model: Any,
    tokenizer: Any,
    direct_prompt: str,
    target_token_id: Optional[int] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:
    # computes attribution on the biased prompt when the model answers directly without any CoT reasoning

    # Baseline 1 - Direct Answer Attribution without intermediate reasoning
    res = attribute_input_x_gradient(
        model, tokenizer, direct_prompt, target_token_id=target_token_id, device=device
    )
    res["method"] = "direct_ixg"
    return res


# Mapping each method name to its implementation
ATTRIBUTION_REGISTRY = {
    "direct_ixg": attribute_direct_answer,
    "raw_attention": attribute_raw_attention,
    "attention_rollout": attribute_attention_rollout,
    "input_x_gradient": attribute_input_x_gradient,
    "integrated_gradients": attribute_integrated_gradients,
    "gradient_shap": attribute_gradient_shap,
}


def compute_attribution(
    method_name: str,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_token_id: Optional[int] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:
    
    # Master function using ATTRIBUTION_REGISTRY

    if method_name not in ATTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown attribution method: {method_name}. Options: {list(ATTRIBUTION_REGISTRY.keys())}")
    
    fn = ATTRIBUTION_REGISTRY[method_name]
    return fn(model, tokenizer, prompt_text, target_token_id=target_token_id, device=device, **kwargs)
