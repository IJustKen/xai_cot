import os
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"


def plot_has_comparison(df: pd.DataFrame, out_path: str = "results/figures/fig1_has_comparison.png") -> None:
    set_plot_style()
    plt.figure(figsize=(9, 5))

    order = [
        "direct_ixg",
        "raw_attention",
        "attention_rollout",
        "input_x_gradient",
        "integrated_gradients",
        "gradient_shap",
    ]
    present_order = [m for m in order if m in df["method"].unique()]
    if not present_order:
        present_order = list(df["method"].unique())

    stats = df.groupby("method")["has"].agg(["mean", "std", "count"]).reindex(present_order)
    means = stats["mean"]
    errors = stats["std"] / (stats["count"] ** 0.5)

    x_positions = range(len(present_order))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    plt.bar(
        x_positions,
        means,
        yerr=errors,
        capsize=5,
        color=colors[: len(present_order)],
        edgecolor="#333333",
        alpha=0.85,
    )

    labels = [
        m.replace("_", " ").title().replace("Ixg", "IxG").replace("Shap", "SHAP")
        for m in present_order
    ]
    plt.xticks(x_positions, labels, rotation=20, ha="right")
    plt.ylabel("Hint Attribution Score (HAS)")
    plt.xlabel("Attribution Method")
    plt.title("Hint Attribution Score (HAS) Across XAI Methods")
    plt.ylim(0, 1.0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure to {out_path}")


def plot_comprehensiveness(df: pd.DataFrame, out_path: str = "results/figures/fig2_comprehensiveness.png") -> None:
    set_plot_style()
    plt.figure(figsize=(9, 5))

    order = [
        "direct_ixg",
        "raw_attention",
        "attention_rollout",
        "input_x_gradient",
        "integrated_gradients",
        "gradient_shap",
    ]
    present_order = [m for m in order if m in df["method"].unique()]
    if not present_order:
        present_order = list(df["method"].unique())

    stats = df.groupby("method")["comprehensiveness"].agg(["mean", "std", "count"]).reindex(present_order)
    means = stats["mean"]
    errors = stats["std"] / (stats["count"] ** 0.5)

    x_positions = range(len(present_order))
    colors = ["#3274A1", "#E1812C", "#3A923A", "#C03D3E", "#9672BE", "#8C564B"]

    plt.bar(
        x_positions,
        means,
        yerr=errors,
        capsize=5,
        color=colors[: len(present_order)],
        edgecolor="#333333",
        alpha=0.85,
    )

    labels = [
        m.replace("_", " ").title().replace("Ixg", "IxG").replace("Shap", "SHAP")
        for m in present_order
    ]
    plt.xticks(x_positions, labels, rotation=20, ha="right")
    plt.ylabel("Confidence Drop (Top 10% Erased)")
    plt.xlabel("Attribution Method")
    plt.title("Erasure Comprehensiveness Test")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure to {out_path}")


def plot_ig_sensitivity(sens_df: pd.DataFrame, out_path: str = "results/figures/fig3_ig_sensitivity.png") -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Approximation error vs steps
    for base_type in sens_df["baseline_type"].unique():
        sub = sens_df[sens_df["baseline_type"] == base_type]
        grouped = sub.groupby("m_steps")["approx_error"].mean()
        axes[0].plot(grouped.index, grouped.values, marker="o", label=f"Baseline: {base_type}")

    axes[0].set_title("IG Approximation Error vs. Steps")
    axes[0].set_xlabel("Riemann Steps (m)")
    axes[0].set_ylabel("Approximation Error |Sum - Delta|")
    axes[0].legend()

    # Panel 2: Stability vs steps
    for base_type in sens_df["baseline_type"].unique():
        sub = sens_df[sens_df["baseline_type"] == base_type]
        grouped = sub.groupby("m_steps")["stability_spearman"].mean()
        axes[1].plot(grouped.index, grouped.values, marker="s", label=f"Baseline: {base_type}")

    axes[1].set_title("IG Score Stability (Spearman rho)")
    axes[1].set_xlabel("Riemann Steps (m)")
    axes[1].set_ylabel("Rank Correlation with Reference")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure to {out_path}")


def plot_target_ablation(ablation_df: pd.DataFrame, out_path: str = "results/figures/fig4_target_ablation.png") -> None:
    set_plot_style()
    plt.figure(figsize=(6, 5))

    means = [
        ablation_df["final_answer_has"].mean(),
        ablation_df["reasoning_aggregated_has"].mean(),
    ]
    stds = [
        ablation_df["final_answer_has"].std() / (len(ablation_df) ** 0.5),
        ablation_df["reasoning_aggregated_has"].std() / (len(ablation_df) ** 0.5),
    ]
    labels = ["Final Answer Token", "Aggregated Reasoning"]
    colors = ["#4C72B0", "#55A868"]

    plt.bar([0, 1], means, yerr=stds, capsize=6, color=colors, edgecolor="#333333", alpha=0.85, width=0.5)
    plt.xticks([0, 1], labels)
    plt.ylabel("Hint Attribution Score (HAS)")
    plt.title("Target Conditioning Ablation\n(Rationale Dilution Test)")
    plt.ylim(0, 1.0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure to {out_path}")
