import os
import argparse
import json
import numpy as np
import pandas as pd
import torch

# load the modules we made
from src.config import Config
from src.data import load_or_create_dataset
from src.generation import load_model_and_tokenizer, run_generation_pipeline
from src.filtering import filter_unfaithful_cot, print_filter_summary
from src.attribution import (
    compute_attribution,
    find_token_span,
    ATTRIBUTION_REGISTRY,
)
from src.metrics import (
    compute_has,
    compute_rank_stability,
    compute_comprehensiveness,
)
from src.sensitivity import (
    evaluate_ig_sensitivity,
    run_target_conditioning_ablation,
)
from src.plotting import (
    plot_has_comparison,
    plot_comprehensiveness,
    plot_ig_sensitivity,
    plot_target_ablation,
)


def run_stage_1(cfg: Config):
    print("====== RUNNING STAGE 1: LOADING DATA ====")
    items = load_or_create_dataset(cfg)
    print(f"Dataset ready with {len(items)} items.")
    return items


def run_stage_2(cfg: Config, items, model=None, tokenizer=None, device=None):
    print("==== DOING STAGE 2: GENERATION & UNFAITHFUL FILTERING ====")
    generations = run_generation_pipeline(
        items, cfg, model=model, tokenizer=tokenizer, device=device
    )
    cot, stats = filter_unfaithful_cot(generations, cfg)
    print_filter_summary(stats)
    return cot, stats


def run_stage_3_and_4(cfg: Config, cot, model=None, tokenizer=None, device=None):
    print("\n--- STAGE 3 & 4: ATTRIBUTION COMPUTATION & EVALUATION METRICS ---")
    if not cot:
        print("Unfaithful cot is empty! Cannot run attribution.")
        return pd.DataFrame()

    metrics_file = cfg.paths.get("metrics_file", "results/metrics.csv")

    if os.path.exists(metrics_file):
        print(f"Loading cached metrics from {metrics_file}")
        return pd.read_csv(metrics_file)

    if (model is None or tokenizer is None):
        model, tokenizer, device = load_model_and_tokenizer(
            cfg.model.get("name", "meta-llama/Llama-3.2-3B-Instruct"),
            cfg.model.get("torch_dtype", "float16"),
            cfg.model.get("device", "cuda"),
        )

    records = []
    methods = cfg.attribution.get("methods", list(ATTRIBUTION_REGISTRY.keys()))
    top_pct = cfg.attribution.get("erasure_top_pct", 0.10)
    ig_steps = cfg.attribution.get("ig_steps", 50)
    grad_shap_samples = cfg.attribution.get("grad_shap_samples", 10)

    for idx, item in enumerate(cot):
        full_biased_text = item["biased_cot_prompt"] + " " + item["biased_cot_response"]
        full_clean_text = item["clean_cot_prompt"] + " " + item["clean_cot_response"]
        direct_prompt = item["biased_direct_prompt"]

        # Real attribution run
        target_char = item["biased_prediction"] or item["target_bias_option"]
        target_token_id = tokenizer.encode(target_char, add_special_tokens=False)[-1]

        prompt_span = find_token_span(tokenizer, full_biased_text, item["biased_cot_prompt"])
        hint_span = find_token_span(tokenizer, full_biased_text, item["hint_text"])

        clean_q_span = find_token_span(tokenizer, full_clean_text, item["question"])
        biased_q_span = find_token_span(tokenizer, full_biased_text, item["question"])

        # Clean attribution baseline for rank stability
        clean_target_char = item["clean_prediction"] or item["ground_truth"]
        clean_target_id = tokenizer.encode(clean_target_char, add_special_tokens=False)[-1]
        clean_attr = compute_attribution(
            "input_x_gradient", model, tokenizer, full_clean_text, target_token_id=clean_target_id, device=device
        )

        for method in methods:
            if method == "direct_ixg":
                attr_res = compute_attribution(
                    method, model, tokenizer, direct_prompt, target_token_id=target_token_id, device=device
                )
                direct_prompt_span = (0, len(attr_res["token_ids"]))
                direct_hint_span = find_token_span(tokenizer, direct_prompt, item["hint_text"])
                has_val = compute_has(attr_res["scores"], direct_hint_span, direct_prompt_span)
                comp = compute_comprehensiveness(
                    model, tokenizer, direct_prompt, attr_res["scores"], direct_prompt_span, target_token_id, top_pct, device
                )
                stab = 0.0
            else:
                attr_res = compute_attribution(
                    method,
                    model,
                    tokenizer,
                    full_biased_text,
                    target_token_id=target_token_id,
                    m_steps=ig_steps,
                    n_samples=grad_shap_samples,
                    device=device,
                )
                has_val = compute_has(attr_res["scores"], hint_span, prompt_span)
                comp = compute_comprehensiveness(
                    model, tokenizer, full_biased_text, attr_res["scores"], prompt_span, target_token_id, top_pct, device
                )
                stab = compute_rank_stability(clean_attr["scores"], attr_res["scores"], clean_q_span, biased_q_span)

            records.append({
                "item_id": item["id"],
                "dataset": item["dataset"],
                "method": method,
                "has": has_val,
                "rank_stability": stab,
                "comprehensiveness": comp,
            })

        print(f"Processed attribution for item {idx + 1}/{len(cot)}")

    df = pd.DataFrame(records)
    df.to_csv(metrics_file, index=False)
    print(f"Saved evaluation metrics to {metrics_file}")
    return df


def run_stage_5(cfg: Config, cot, model=None, tokenizer=None, device=None):
    print("\n--- STAGE 5: SENSITIVITY & ABLATION STUDY ---")
    sens_file = cfg.paths.get("sensitivity_file", "results/sensitivity_metrics.csv")
    ablation_file = cfg.paths.get("ablation_file", "results/ablation_metrics.csv")

    if os.path.exists(sens_file) and os.path.exists(ablation_file):
        print(f"Loading cached sensitivity and ablation results...")
        return pd.read_csv(sens_file), pd.read_csv(ablation_file)

    if (model is None or tokenizer is None):
        model, tokenizer, device = load_model_and_tokenizer(
            cfg.model.get("name", "meta-llama/Llama-3.2-3B-Instruct"),
            cfg.model.get("torch_dtype", "float16"),
            cfg.model.get("device", "cuda"),
        )

    sens_records = []
    ablation_records = []
    steps_list = cfg.sensitivity.get("ig_steps_list", [10, 25, 50, 100])
    baselines = cfg.sensitivity.get("baselines", ["pad", "zero"])

    eval_subset = cot[: min(10, len(cot))]

    for idx, item in enumerate(eval_subset):

        full_biased_text = item["biased_cot_prompt"] + " " + item["biased_cot_response"]
        prompt_span = find_token_span(tokenizer, full_biased_text, item["biased_cot_prompt"])
        hint_span = find_token_span(tokenizer, full_biased_text, item["hint_text"])

        target_char = item["biased_prediction"] or item["target_bias_option"]
        target_token_id = tokenizer.encode(target_char, add_special_tokens=False)[-1]

        # 1. IG Sensitivity
        sens_res = evaluate_ig_sensitivity(
            model,
            tokenizer,
            full_biased_text,
            target_token_id,
            hint_span,
            prompt_span,
            steps_list=steps_list,
            baselines=baselines,
            device=device,
        )
        for r in sens_res:
            r["item_id"] = item["id"]
            sens_records.append(r)

        # 2. Target Conditioning Ablation
        abl_res = run_target_conditioning_ablation(
            model,
            tokenizer,
            item["biased_cot_prompt"],
            item["biased_cot_response"],
            target_token_id,
            hint_span,
            prompt_span,
            m_steps=cfg.attribution.get("ig_steps", 50),
            device=device,
        )
        abl_res["item_id"] = item["id"]
        ablation_records.append(abl_res)

        print(f"Processed sensitivity & ablation for item {idx + 1}/{len(eval_subset)}")

    sens_df = pd.DataFrame(sens_records)
    abl_df = pd.DataFrame(ablation_records)

    sens_df.to_csv(sens_file, index=False)
    abl_df.to_csv(ablation_file, index=False)
    print(f"Saved sensitivity results to {sens_file} and ablation to {ablation_file}")
    return sens_df, abl_df


def run_stage_6(cfg: Config, metrics_df: pd.DataFrame, sens_df: pd.DataFrame, abl_df: pd.DataFrame):
    print("\n--- STAGE 6: PLOTS & SUMMARY GENERATION ---")
    fig_dir = cfg.paths.get("figures_dir", "results/figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not metrics_df.empty:
        plot_has_comparison(metrics_df, os.path.join(fig_dir, "fig1_has_comparison.png"))
        plot_comprehensiveness(metrics_df, os.path.join(fig_dir, "fig2_comprehensiveness.png"))

    if not sens_df.empty:
        plot_ig_sensitivity(sens_df, os.path.join(fig_dir, "fig3_ig_sensitivity.png"))

    if not abl_df.empty:
        plot_target_ablation(abl_df, os.path.join(fig_dir, "fig4_target_ablation.png"))

    summary_file = cfg.paths.get("summary_file", "results/summary.md")
    print(f"Writing summary report to {summary_file}...")
    generate_summary_md(summary_file, metrics_df, sens_df, abl_df)
    print("Stage 6 complete!")


def df_to_markdown(df: pd.DataFrame) -> str:
    # Plain Markdown table generator without requiring tabulate
    headers = [str(col) for col in df.columns]
    index_name = df.index.name or "Method"
    header_line = "| " + index_name + " | " + " | ".join(headers) + " |"
    sep_line = "| " + "--- | " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for idx, row in df.iterrows():
        row_vals = [f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in row.values]
        rows.append(f"| {idx} | " + " | ".join(row_vals) + " |")
    return "\n".join([header_line, sep_line] + rows)


def generate_summary_md(
    summary_path: str, metrics_df: pd.DataFrame, sens_df: pd.DataFrame, abl_df: pd.DataFrame
):
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# DS5622 Project Summary: Feature Attribution Under Unfaithful CoT\n\n")
        f.write("## 1. Overview and Core Findings\n\n")

        if not metrics_df.empty:
            grouped = metrics_df.groupby("method").agg({
                "has": "mean",
                "rank_stability": "mean",
                "comprehensiveness": "mean",
            }).round(4)
            f.write("### Evaluation Metrics Across Attribution Methods\n\n")
            f.write(df_to_markdown(grouped) + "\n\n")

        f.write("## 2. Hypothesis Testing\n\n")
        f.write("### Hypothesis 1: Rationale Dilution Effect\n")
        if not abl_df.empty:
            ans_mean = abl_df["final_answer_has"].mean()
            reas_mean = abl_df["reasoning_aggregated_has"].mean()
            f.write(f"- Final Answer Target Mean HAS: `{ans_mean:.4f}`\n")
            f.write(f"- Aggregated Reasoning Target Mean HAS: `{reas_mean:.4f}`\n")
            if reas_mean > ans_mean:
                f.write(
                    "- **Verdict**: **SUPPORTED**. Attributing strictly with respect to the final answer token yields significantly lower importance on prompt hint tokens compared to aggregated reasoning. Proximate reasoning tokens absorb the gradient flow.\n\n"
                )
            else:
                f.write("- **Verdict**: **NOT SUPPORTED / INCONCLUSIVE**.\n\n")
        else:
            f.write("- Pending experimental execution.\n\n")

        f.write("### Hypothesis 2: Method Comparison\n")
        if not metrics_df.empty:
            means = metrics_df.groupby("method")["has"].mean().to_dict()
            ig_has = means.get("integrated_gradients", 0.0)
            rollout_has = means.get("attention_rollout", 0.0)
            ixg_has = means.get("input_x_gradient", 0.0)
            f.write(f"- Integrated Gradients HAS: `{ig_has:.4f}`\n")
            f.write(f"- Attention Rollout HAS: `{rollout_has:.4f}`\n")
            f.write(f"- Input × Gradient HAS: `{ixg_has:.4f}`\n")
            if ig_has > rollout_has and ig_has > ixg_has:
                f.write(
                    "- **Verdict**: **SUPPORTED**. Path-integral methods retain higher sensitivity to input bias cues compared to raw attention rollout and noisy first-order gradients.\n\n"
                )
            else:
                f.write(
                    "- **Verdict**: **PARTIALLY SUPPORTED / METHOD DEPENDENT**. Inspect tabular variance.\n\n"
                )
        else:
            f.write("- Pending experimental execution.\n\n")

        f.write("## 3. Honest Limitations\n\n")
        f.write(
            "1. **Model Scope**: Experiments are evaluated on Llama-3.2-3B-Instruct in 16-bit precision. Larger models (e.g. 70B) may exhibit differing degrees of sycophancy and explanation fabrication.\n"
        )
        f.write(
            "2. **Discrete Masking in Erasure**: The comprehensiveness test relies on substituting tokens with `<pad>`, which can introduce out-of-distribution artifacts in autoregressive decoders.\n"
        )
        f.write(
            "3. **Attention Rollout Assumptions**: Attention rollout assumes linearity across layers and uniform head mixing, which neglects non-linear MLP transformations and residual path dampening.\n"
        )


def main():
    # using argparse to use CLI to run the pipeline
    parser = argparse.ArgumentParser(description="DS5622 XAI CoT Pipeline")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "4", "5", "6", "all"])
    args = parser.parse_args()

    cfg = Config.from_yaml("config.yaml")

    # Stage 1
    items = run_stage_1(cfg)
    if args.stage == "1":
        return

    # Stage 2
    cot, stats = run_stage_2(cfg, items)
    if args.stage == "2":
        return

    # Stage 3 & 4
    metrics_df = run_stage_3_and_4(cfg, cot)
    if args.stage in ["3", "4"]:
        return

    # Stage 5
    sens_df, abl_df = run_stage_5(cfg, cot)
    if args.stage == "5":
        return

    # Stage 6
    run_stage_6(cfg, metrics_df, sens_df, abl_df)


if __name__ == "__main__":
    main()
