"""
Recomputes every p-value in the report from the saved result files.

Usage:  python compute_stats.py [folder_with_result_files]

All tests are two-sided and come from scipy.stats with default settings:
  - mannwhitneyu : two independent groups (unfaithful vs no-flip)
  - wilcoxon     : paired values on the same examples, or one sample against zero
No correction for multiple comparisons is applied.
"""
import sys, os
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

D = sys.argv[1] if len(sys.argv) > 1 else "."
def load(name): return pd.read_json(os.path.join(D, name), lines=True)
def task_of(i): return i.rsplit("_", 1)[0]
def show(label, test, n, p): print(f"{label:62s} {test:13s} n={n:<8s} p={p:.2g}")
def mw(label, a, b): show(label, "Mann-Whitney", f"{len(a)}/{len(b)}", mannwhitneyu(a, b).pvalue)
def wx(label, a, b=None): show(label, "Wilcoxon", str(len(a)), (wilcoxon(a, b) if b is not None else wilcoxon(a)).pvalue)

p2 = load("phase2_results.jsonl"); p2["task"] = p2.id.map(task_of)
U, N = p2[p2.label == "unfaithful"], p2[p2.label == "no_flip"]

print("== Final answer: unfaithful vs no-flip (Table 1, Sec. 4.5) ==")
for c, name in [("ixg_HAS_ratio", "Input x Gradient HAS ratio"), ("ig_HAS_ratio", "IG HAS ratio"),
                ("attn_HAS_ratio", "Attention HAS ratio"), ("ixg_signed_hint", "Input x Gradient signed score"),
                ("ig_signed_hint", "IG signed score"), ("ixg_reasoning_HAS_ratio", "Target ablation: reasoning-sum HAS ratio")]:
    if c in p2.columns: mw(name, U[c], N[c])
    else: print(f"{name:62s} (column {c} not in phase2_results.jsonl)")

gs = load("phase3_gradshap.jsonl"); gu, gn = gs[gs.label == "unfaithful"], gs[gs.label == "no_flip"]
mw("GradientSHAP HAS ratio", gu.gs_HAS_ratio, gn.gs_HAS_ratio)
mw("GradientSHAP signed score", gu.gs_signed, gn.gs_signed)

print("\n== Interventions on the hint at the final answer (Table 2, Sec. 5) ==")
er = load("phase3_erasure.jsonl"); cf = load("phase3_hint_counterfactual.jsonl")
for lab in ("unfaithful", "no_flip"):
    wx(f"{lab}: pad erasure vs 0", er[er.label == lab].drop_hint_only)
    wx(f"{lab}: random-token replacement vs 0", cf[cf.label == lab].drop_randtok)
    wx(f"{lab}: letter swap vs 0", cf[cf.label == lab].drop_swap)

print("\n== Comprehensiveness, top-10% erasure (Sec. 4.5, App. D) ==")
m = er.merge(gs[["id", "gs_drop"]], on="id")
for lab in ("unfaithful", "no_flip"):
    s = m[m.label == lab]
    for c, name in [("drop_ixg", "Input x Gradient"), ("drop_ig", "IG"), ("drop_attn", "Attention"),
                    ("drop_rollout", "Rollout"), ("gs_drop", "GradientSHAP")]:
        wx(f"{lab}: {name} vs random", s[c], s.drop_random)
    wx(f"{lab}: Attention vs IG", s.drop_attn, s.drop_ig)
    wx(f"{lab}: GradientSHAP vs IG", s.gs_drop, s.drop_ig)
    wx(f"{lab}: Input x Gradient vs IG", s.drop_ixg, s.drop_ig)

print("\n== Divergence point (Sec. 4.3, App. D) ==")
dv = load("phase3_divergence.jsonl"); dv = dv[dv.label != "flip_acknowledged"]
b = dv.merge(p2[["id", "ixg_HAS_ratio", "ig_HAS_ratio", "attn_HAS_ratio"]], on="id")
wx("hint erasure vs random erasure (all)", dv.drop_hint, dv.drop_rand)
late, early = dv[dv.div_pos > 10], dv[dv.div_pos <= 10]
wx("hint erasure vs random erasure (divergence after token 10)", late.drop_hint, late.drop_rand)
wx("hint erasure vs random erasure (token 10 or earlier)", early.drop_hint, early.drop_rand)
for k, name in [("ixg", "Input x Gradient"), ("ig", "IG"), ("attn", "Attention")]:
    wx(f"{name} HAS ratio: divergence point vs final answer", b[f"{k}_div"], b[f"{k}_HAS_ratio"])
wx("Input x Gradient signed score at divergence vs 0", dv.ixg_signed)
mw("divergence position, unfaithful vs no-flip",
   dv[dv.label == "unfaithful"].div_pos, dv[dv.label == "no_flip"].div_pos)

print("\n== Direct answer vs with reasoning (Sec. 4.4, App. C) ==")
di = load("phase3_direct.jsonl"); di = di[di.label != "flip_acknowledged"]
d = di.merge(p2[["id", "task", "ixg_HAS_ratio", "ig_HAS_ratio", "attn_HAS_ratio"]], on="id")
for k, name in [("ixg", "Input x Gradient"), ("ig", "IG"), ("attn", "Attention")]:
    wx(f"{name}: direct vs with reasoning (all)", d[f"{k}_direct"], d[f"{k}_HAS_ratio"])
s = d[d.task == "sports_understanding"]
wx("Input x Gradient: direct vs with reasoning (Sports)", s.ixg_direct, s.ixg_HAS_ratio)
wx("IG: direct vs with reasoning (Sports)", s.ig_direct, s.ig_HAS_ratio)

print("\n== Rank stability (Sec. 4.5) ==")
rs = load("phase3_rank_stability.jsonl"); ru, rn = rs[rs.label == "unfaithful"], rs[rs.label == "no_flip"]
mw("Input x Gradient Spearman rho", ru.rho_ixg, rn.rho_ixg)
mw("Attention Spearman rho", ru.rho_attn, rn.rho_attn)
