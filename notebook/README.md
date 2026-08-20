# Notebook index

Notebooks are ordered by their appearance in the paper. Each notebook is
self-contained and Colab-runnable. To reproduce a specific paper result,
open the corresponding notebook and run all cells.

## Notebooks

| Notebook | Paper section | What it does |
|---|---|---|
| `01_cider_reimplementation.ipynb` | Sec. 3.3, Supplementary A | Reimplements CIDER's three-stage framework and validates against Zhang et al.'s reported F1 on the CO → WT pair (Zhang et al. 2025, Information Fusion 117:102816). |
| `02_llm_paradigm_matrix.ipynb` | Sec. 4.1–4.2 | Main LLM matrix: 3 backbones × 4 demonstration-selection methods × 4 cross-domain pairs × 3 seeds. |
| `03_statistical_analysis.ipynb` | Sec. 4.1, Table 5 | Wilcoxon signed-rank tests with Bonferroni correction across the LLM matrix. |
| `04_dader_baseline.ipynb` | Sec. 4.3, Table 6 | DADER domain-adversarial baseline (zero target labels). |
| `05_schema_poverty_ablation_llm.ipynb` | Sec. 4.6, Table 12 (LLM column) | LLM K=2 F1 across the mutual-information-ranked attribute-drop ladder. |
| `06_schema_poverty_ablation_ditto.ipynb` | Sec. 4.6, Table 12 (SFT column) | Ditto warm-start K=100 across the same ladder. Also contains contamination-test cells (Table 13). |
| `07_triangle_violation_test.ipynb` | Sec. 4.7, Table 14 | Triangle-violation rate on WDC-Watches multi-record clusters. |
| `08_llm_lora_finetune.ipynb` | Sec. 4.4.1, Table 8 | LLM LoRA K=100 fine-tuning on Llama-3.1-8B across P1–P4. |
| `09_matchgpt_k10_sub_analysis.ipynb` | Sec. 4.2 sub-analysis | K=2 vs K=10 sub-analysis on Llama-3.3-70B (per-cell metrics referenced from §4.2). |

## Note on Ditto SFT baselines

Ditto from-scratch K=100, Ditto warm-start K=100, and Ditto zero-shot results
were produced using Ditto's own training script (megagonlabs/ditto) after
data preparation via `src/data/` and warm-start helpers in
`src/experiments/ditto_warmstart.py`. A minimal reproduction sequence:

```bash
# Prepare data
python -m src.data.download Structured/Walmart-Amazon
python -m src.data.convert_to_ditto Structured/Walmart-Amazon
python -m src.data.make_splits --k 100 --seed 42 Structured/Walmart-Amazon

# Train (Ditto standard trainer)
python ditto/train_ditto.py \
    --task Structured/Walmart-Amazon-k100-s42 \
    --n_epochs 15 --batch_size 32 --lm roberta
```

Per-cell metrics are stored under `results/runs/ditto-warmstart/`,
`results/runs/ditto-vanilla-baseline/`, and `results/runs/ditto-zeroshot/`.

## Prerequisites

- Python 3.10+
- Google Colab account with GPU access (for SFT and LoRA notebooks)
- DeepInfra API key (for LLM prompting notebooks)
- Hugging Face token (for downloading Llama models in the LoRA notebook)
- Environment variable `REPO_ROOT` pointing to the repo root

Set API keys via a `.env` file in the repo root (see `.env.example`), or via
Colab secrets.
