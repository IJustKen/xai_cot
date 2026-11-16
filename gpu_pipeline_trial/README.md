# DS5622 Explainable AI Course Project
## Investigating Feature Attribution Methods When Chain-of-Thought Reasoning is Unfaithful

**Team Members**:
- Akshat Nair 142301038
- Jeeva Jayaprakash 142301012

This repo contains the experimental codebase for investigating how typical post-hoc feature attribution methods behave when Large Language Models (LLMs) output unfaithful Chain-of-Thought (CoT) explanations due to injected prompt biases.

---

## 1. Repository Structure

```text
xai_cot/
|-- README.md                    # Overall guide
|-- requirements.txt             # Python package dependencies
|-- config.yaml                  # Configuration file to avoid hardcoding paths and other parameters
|-- run_pipeline.py              # Main CLI runner for the project
|-- notebooks/
|   |-- 01_data_and_filtering.ipynb    # Stage 1 & 2: Dataset curation, inference and unfaithfulness filtering
|   |-- 02_attribution.ipynb           # Stage 3: Attribution computation across methods
|   |-- 03_evaluation.ipynb            # Stage 4: HAS, Rank Stability and Comprehensiveness
|   |__ 04_sensitivity_ablation.ipynb  # Stage 5 & 6: Sensitivity, Target Conditioning and Summary
|-- src/
|   |-- __init__.py
|   |-- config.py                # Dataclass loader
|   |-- data.py                  # Curation and pairing for BBH (Sports Understanding, Navigate, Web of Lies) & BBQ
|   |-- generation.py            # Infering the LLM
|   |-- filtering.py             # Unfaithful CoT detector (flip check & hint leakage check)
|   |-- attribution.py           # Attention, Rollout, IxG, IG, GradSHAP
|   |-- metrics.py               # HAS, Spearman rho, Erasure
|   |-- sensitivity.py           # IG step sensitivity & target conditioning ablation
|   |__ plotting.py              # Matplotlib figures
|-- data/
|   |-- raw/                     # Downloaded task splits
|   |__ processed/               # Paired clean and biased prompts
|-- results/
    |-- generations.jsonl        # Saved model completions
    |-- unfaithful_cot.jsonl     # Filtered unfaithful evaluation set
    |-- metrics.csv              # Attribution metrics results
    |-- sensitivity_metrics.csv  # IG step sensitivity table
    |-- ablation_metrics.csv     # Ablation table
    |-- figures/                 # Plots
    |__ summary.md               # Summary and hypothesis verdicts
```

---

## 2. Environment Setup
We mostly ran the codebase on Colab GPU (~16GB)

### Requirements
```bash
pip install -r requirements.txt
```

### Hugging Face Access Token
Set your Hugging Face user access token in a `.env` file (refer to `.env.example` for the format):

In Colab, you can also store it under notebook *Secrets* with the name `HF_TOKEN`.

---

## 3. How to Run

### Fast Smoke Test (5 Examples)
Smoke test to quickly verify all stages are functional:
```bash
python run_pipeline.py
```

### Execution
You may execute each stage independently. All intermediate results are saved in `data/processed/` and `results/`. We did this because Colab runtime gets disconnected often and we kept losing results.

Stage 1: Create & Pair Dataset (300 pairs)
   ```bash
   python run_pipeline.py --stage 1
   ```
Stage 2: Model Inference & Unfaithful Filtering
   ```bash
   python run_pipeline.py --stage 2
   ```
   *To see the flip rate, hint leakage count and total unfaithful cases.*

Stage 3 & 4: Attribution Computation & Metrics
   ```bash
   python run_pipeline.py --stage 3
   ```
Stage 5: Sensitivity Analysis & Ablation
   ```bash
   python run_pipeline.py --stage 5
   ```
Stage 6: Plotting & Final Summary Report
   ```bash
   python run_pipeline.py --stage 6
   ```

### For the Full Run
```bash
python run_pipeline.py --stage all
```

