# Cross-Domain Entity Resolution Paradigm Choice

Reproducibility artifact for the paper:

> "Paradigm Choice for Cross-Domain Entity Resolution under Attribute Sparsity:
> A Quality, Cost, Latency, Consistency Study."
> Submitted to the PVLDB Vol 20 Experiment, Analysis & Benchmark track.

This repository contains the source code, per-cell measurements, and
supplementary appendices required to reproduce every result reported in
the paper. 

## Repository layout

```
.
├── src/
│   ├── analysis/          # Statistical analysis, figures, cost/latency
│   ├── data/              # Data ingestion and preprocessing
│   ├── experiments/       # Runner scripts (Ditto, DADER, LoRA)
│   └── utils/             # Ditto log parser and helpers
├── notebook/              # Colab-runnable experiment notebooks (see notebook/README.md)
├── results/
│   ├── runs/              # Per-cell metrics.json for every table
│   └── figures/           # Generated PDF and PNG figures
├── docs/
│   └── SUPPLEMENTARY.pdf  # Supplementary appendices
├── config/
│   └── config.py          # Path resolution and hyperparameter defaults
├── docker-compose.yml     # Reproducible analysis environment
├── requirements.txt       # Python dependencies
└── README.md              # (this file)
```

## Quick start

### Option 1: Docker (recommended for reproducibility)

```bash
docker compose up -d
docker compose exec analysis bash
# Then, inside the container:
export REPO_ROOT=/home/jovyan/work
python -c "import config.config; print('OK')"
```

### Option 2: Local Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export REPO_ROOT=$(pwd)
python -c "import config.config; print('OK')"
```

## Reproducing paper results

Each paper section maps to one or more notebooks in `notebook/`. See
`notebook/README.md` for the full index. 

## Data sources

Public benchmark datasets are downloaded from their upstream sources:

- Magellan corpus (Walmart-Amazon, Abt-Buy, DBLP-ACM, Amazon-Google):
  https://github.com/anhaidgroup/deepmatcher
- WDC-Products (Computers, Watches):
  http://webdatacommons.org/largescaleproductcorpus/

Download scripts are in `src/data/`.

## Environment variables

- `REPO_ROOT` — path to this repo (used by `config/config.py`)
- `DEEPINFRA_API_KEY` — for LLM inference notebooks
- `HF_TOKEN` — for downloading Llama models in the LoRA fine-tuning notebook

Copy `.env.example` to `.env` and fill in.

## Reproducibility notes

- All LLM cells report **three-seed** measurements (seeds 42, 123, 456).
- Per-cell raw measurements are in `results/runs/**/metrics.json`. Filenames
  encode `(pair, method, seed)`.
- CIDER reimplementation: the CIDER paper does not release source code.
  Our reimplementation follows Algorithm 1 and Equations 1–9 from the
  paper. A side-by-side comparison of our Python implementation with the
  paper's pseudocode is in `docs/SUPPLEMENTARY.pdf` Section A.1.
- One deviation from CIDER: pure top-K demonstration selection is
  replaced with a stratified top-(K/2) positive + top-(K/2) negative
  variant to handle the >99%-negative candidate pool that occurs on
  extreme cross-domain source–target combinations. Justified in
  `docs/SUPPLEMENTARY.pdf` Section A.2.

## License

MIT
