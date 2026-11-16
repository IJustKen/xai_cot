import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_answer_letter(text: str) -> Optional[str]:
    # Look for standard CoT conclusion first
    # Uses regex to find the model's declared final answer letter (like example: from "Therefore, the answer is (B)", get B).

    patterns = [    # this was done with the help of AI
        r"(?:therefore[,\s]+)?(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-D])\)?",    # therefore the .. answer is.. option
        r"(?:conclude[,\s]+that\s+)?\(?([A-D])\)?\s+is\s+(?:the\s+)?(?:correct|best)\s+answer",     # conclude that .. option .. correct/best answer
        r"\(([A-D])\)",     # in case it directly outputs the option
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    print(f"For TEXT:{text} --> Nothing matched")

    return None


def get_hf_token():
    return os.environ.get("HF_TOKEN")


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "float16",
    device: str = "cuda",
) -> Tuple[Any, Any, str]:

    # Loads the Llama-3.2-3B model and tokenizer in fp16/bf16
    # Automatically falls back to CPU if no GPU is available
    
    # AI was used to help debug some of this

    token = get_hf_token()
    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    target_device = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"    # just to make it runnable on cpu too
    print(f"Loading {model_name} on {target_device} ({dtype})....")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if target_device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            token=token,
        ).to("cpu")

    model.eval()
    return model, tokenizer, target_device


def generate_single(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
    device: str = "cuda",
) -> Tuple[str, str]:

    # Runs autoregressive generation 
    # (do_sample=False, temperature=0.0) for reproducible outputs
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0, prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return gen_text, tokenizer.decode(output_ids[0], skip_special_tokens=False)


def run_generation_pipeline(
    items: List[Dict[str, Any]],
    cfg: Any,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:

    # Loops over the dataset, generating: Clean CoT output & prediction, Biased CoT output & prediction
    # and Biased direct output & prediction
    
    out_file = cfg.paths.get("generations_file", "results/generations.jsonl")

    # Resume from cache if already generated (helpful in colab)
    cached_records = {}
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached_records[rec["id"]] = rec
        print(f"Loaded {len(cached_records)} cached generations from {out_file}")

    pending_items = [it for it in items if it["id"] not in cached_records]

    if not pending_items:
        return [cached_records[it["id"]] for it in items]

    if (model is None or tokenizer is None):
        model, tokenizer, device = load_model_and_tokenizer(
            cfg.model.get("name", "meta-llama/Llama-3.2-3B-Instruct"),
            cfg.model.get("torch_dtype", "float16"),
            cfg.model.get("device", "cuda"),
        )

    max_new_tokens = cfg.generation.get("max_new_tokens", 256)
    temp = cfg.generation.get("temperature", 0.0)   # to reproduce well
    do_sample = cfg.generation.get("do_sample", False)

    results = list(cached_records.values())

    print(f"Generating answers for {len(pending_items)} items..")
    with open(out_file, "a", encoding="utf-8") as f_out:
        for idx, it in enumerate(pending_items):
            item_id = it["id"]

            clean_gen, _ = generate_single(
                model, tokenizer, it["clean_cot_prompt"], max_new_tokens, temp, do_sample, device
            )
            pred_clean = parse_answer_letter(clean_gen)

            biased_gen, _ = generate_single(
                model, tokenizer, it["biased_cot_prompt"], max_new_tokens, temp, do_sample, device
            )
            pred_biased = parse_answer_letter(biased_gen)

            direct_gen, _ = generate_single(
                model, tokenizer, it["biased_direct_prompt"], 32, temp, do_sample, device
            )

            pred_direct = parse_answer_letter(direct_gen)

            record = {
                "id": item_id,
                "dataset": it["dataset"],
                "question": it["question"],
                "options": it["options"],
                "ground_truth": it["ground_truth"],
                "target_bias_option": it["target_bias_option"],
                "hint_text": it["hint_text"],
                "clean_cot_prompt": it["clean_cot_prompt"],
                "clean_direct_prompt": it["clean_direct_prompt"],
                "biased_cot_prompt": it["biased_cot_prompt"],
                "biased_direct_prompt": it["biased_direct_prompt"],
                "clean_cot_response": clean_gen,
                "biased_cot_response": biased_gen,
                "biased_direct_response": direct_gen,
                "clean_prediction": pred_clean,
                "biased_prediction": pred_biased,
                "direct_prediction": pred_direct,
            }

            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            results.append(record)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(pending_items):
                print(f"Generated {idx + 1}/{len(pending_items)} items")

    # Sort back by original ID order
    results.sort(key=lambda x: x["id"])
    return results
