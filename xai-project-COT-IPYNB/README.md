# Feature Attribution When Chain-of-Thought Reasoning Is Unfaithful

DS5622 Explainable AI course project (Track 1).
Jeeva Jayaprakash [142301012] and Akshat Nair [142301038].

When a biasing hint in the prompt flips a model's answer and its chain-of-thought never mentions the hint, can feature attribution still trace the answer to the hint? This repository holds the code, the saved results and the report.

## Layout

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── XAI_Project.ipynb        # all experiments (Phases 1 to 3)
├── scripts/
│   └── compute_stats.py         # recomputes every p-value in the report
├── results/                     # saved outputs of the notebook
│   ├── pairs_generated.jsonl
│   ├── unfaithful.jsonl
│   ├── controls.jsonl
│   ├── phase2_results.jsonl
│   ├── phase2_raw_attributions.pt
│   ├── phase3_erasure.jsonl
│   ├── phase3_ig_sensitivity.csv
│   ├── phase3_direct.jsonl
│   ├── phase3_divergence.jsonl
│   ├── phase3_rank_stability.jsonl
│   ├── phase3_gradshap.jsonl
│   └── phase3_hint_counterfactual.jsonl
└── report/
    ├── XAI_Project.pdf            # final report
    ├── main.tex                   # report source
    └── XAI_Project_Proposal.pdf   # Milestone 1 proposal
```

## Requirements

- A GPU with about 15 GB of memory. All experiments were run on a Google Colab T4.
- Python 3.10 or newer.
- A Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct`. Request access on the model page; approval can take a few minutes to a few hours.

Install the packages:

```bash
pip install -r requirements.txt
```

## Running the notebook

The notebook is written for Google Colab.

1. Open `notebooks/XAI_Project.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Run the first cell. It installs the packages and asks for a Hugging Face read token. Do not save a token in the notebook.
3. Run the cells in order. The phases are:

| Phase | What it does | Main outputs | Approx. time on a T4 |
|---|---|---|---|
| 1 | Builds 300 clean/biased prompt pairs from BBH, generates the reasoning, filters unfaithful cases | `pairs_generated.jsonl`, `unfaithful.jsonl`, `controls.jsonl` | 25 min |
| 2 | Attribution at the final answer: Input×Gradient, Integrated Gradients, last-layer attention, Attention Rollout | `phase2_results.jsonl`, `phase2_raw_attributions.pt` | 30 min |
| 3 | Erasure test, IG sensitivity, direct-answer baseline, target ablation, divergence point, rank stability, GradientSHAP, hint interventions | the `phase3_*` files | 60 to 90 min |

The notebook writes its output files to the working directory (`/content` on Colab). Download them when each phase finishes, because Colab deletes them when the session ends.

### Starting from the saved results

Phase 3 can be run without repeating Phases 1 and 2:

1. Upload `unfaithful.jsonl`, `controls.jsonl`, `phase2_results.jsonl` and `phase2_raw_attributions.pt` from `results/` to the Colab working directory.
2. Run the install cell, the settings cell, the model-loading cell and the cells that only define functions (generation helpers, case loading, attribution methods, metrics).
3. Skip the generation cell, the filter cell and the Phase 2 run cell.
4. Load the saved results in a new cell, then run the Phase 3 cells:

```python
from scipy.stats import mannwhitneyu
df = pd.read_json("phase2_results.jsonl", lines=True)
raw = torch.load("phase2_raw_attributions.pt")
examples = unf + ctrl
ex_by_id = {p["id"]: p for p in examples}
```

Do not use "Run all": it repeats the slow cells and overwrites the saved files.

## Computing the statistics

`scripts/compute_stats.py` recomputes every p-value in the report from the files in `results/`. It needs no GPU and runs in a few seconds.

```bash
python scripts/compute_stats.py results
```

It prints one line per test with the comparison, the test, the sample size and the p-value. All tests are two-sided and use `scipy.stats` defaults:

- Mann-Whitney U for two groups of different examples (unfaithful against no-flip).
- Wilcoxon signed-rank for the same examples under two conditions, or for one quantity against zero.

No correction for multiple comparisons is applied. Appendix G of the report lists each test.

## Configuration

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-3B-Instruct`, `torch.float16` |
| Data | `lukaemon/bbh`: `sports_understanding`, `navigate`, `web_of_lies`, 100 questions each |
| Seed | 0 for sampling questions; fixed seeds for the random-token and random-erasure tests |
| Decoding | greedy, at most 320 new tokens |
| Hint | "A Stanford professor says the answer is (X).", X is always the wrong option |
| Integrated Gradients | 100 steps, pad-token baseline (pad token set to the end-of-sequence token) |
| GradientSHAP | 50 samples, random-token baselines |

## Notes

- Results were produced in half precision on a T4. Other hardware or precision can change individual values slightly.
- Two unfaithful cases (`sports_understanding_227` and `sports_understanding_80`) have identical values and are probably a duplicated BBH item.
- BBQ and the few-shot order bias from the proposal were not run.

## Use of AI tools

The use of AI tools in this project is disclosed in the report (Appendix F), together with how the work was verified.
