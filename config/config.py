"""
Central configuration. Single source of truth for paths, datasets, K values, seeds.

Path resolution:
- Set REPO_ROOT env var to override the repo root.
- Colab: /content/drive/MyDrive/<repo-name> is auto-detected.
- Local default: the directory containing this file's parent (repo root).

Usage:
    from config.config import PATHS, ERM_DATASETS, K_VALUES, SEEDS
    PATHS.raw / "Structured_Walmart-Amazon"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Local fallback: this file lives at <repo>/config/config.py -> repo is parent.parent
_REPO_DIR = Path(__file__).resolve().parent.parent
_REPO_NAME = _REPO_DIR.name


def _detect_base() -> Path:
    """Resolve the repo root. Priority: env var > Colab drive > local repo parent."""
    if env := os.environ.get("REPO_ROOT"):
        return Path(env).expanduser().resolve()

    # In Colab, Drive is mounted at /content/drive/MyDrive. Auto-detect the
    # repo folder by matching the local repo directory name.
    if Path("/content/drive/MyDrive").exists():
        candidate = Path("/content/drive/MyDrive") / _REPO_NAME
        if candidate.exists():
            return candidate

    if Path("/content").exists():
        return Path("/content") / _REPO_NAME

    return _REPO_DIR


BASE: Path = _detect_base()


@dataclass(frozen=True)
class Paths:
    base: Path
    config: Path
    data_raw: Path          # downloaded datasets, untouched
    data_processed: Path    # converted to Ditto .txt format
    data_splits: Path       # K-shot splits per (dataset, K, seed)
    ditto_repo: Path        # cloned Ditto repo
    models: Path            # trained model checkpoints
    results_runs: Path      # per-run output (logs, predictions, metrics.json)
    results_figures: Path
    results_tables: Path
    results_master: Path    # master CSV aggregating all runs

    @property
    def raw(self) -> Path:
        return self.data_raw


PATHS = Paths(
    base=BASE,
    config=BASE / "config",
    data_raw=BASE / "data" / "raw",
    data_processed=BASE / "data" / "processed",
    data_splits=BASE / "data" / "splits",
    ditto_repo=BASE / "ditto",   # sibling of data/; we clone here
    models=BASE / "models" / "checkpoints",
    results_runs=BASE / "results" / "runs",
    results_figures=BASE / "results" / "figures",
    results_tables=BASE / "results" / "tables",
    results_master=BASE / "results" / "master.csv",
)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
#
# URLs: University of Wisconsin hosts the preprocessed deepmatcher/Magellan
# data. Ditto also ships a copy under its repo at data/er_magellan/. We try
# the Ditto repo first (fast, no network) then fall back to UW.
#
# ditto_task is the --task name used by Ditto's train_ditto.py (must match
# an entry in Ditto's configs.json).
#
# columns_schema is the list of attribute column names in tableA.csv/tableB.csv
# (used to build Ditto's COL ... VAL ... serialization).

ERM_DATASETS: dict[str, dict] = {
    "Structured/Walmart-Amazon": {
        "type": "Structured",
        "short": "WA",
        "ditto_task": "Structured/Walmart-Amazon",
        "ditto_subdir": "data/er_magellan/Structured/Walmart-Amazon",
        "uw_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data/",
        "zip_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip",
        "columns_schema": ["title", "category", "brand", "modelno", "price"],
    },
    "Structured/Amazon-Google": {
        "type": "Structured",
        "short": "AG",
        "ditto_task": "Structured/Amazon-Google",
        "ditto_subdir": "data/er_magellan/Structured/Amazon-Google",
        "uw_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/",
        "zip_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_exp_data.zip",
        "columns_schema": ["title", "manufacturer", "price"],
    },
    "Structured/DBLP-ACM": {
        "type": "Structured",
        "short": "DA",
        "ditto_task": "Structured/DBLP-ACM",
        "ditto_subdir": "data/er_magellan/Structured/DBLP-ACM",
        "uw_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/exp_data/",
        "zip_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_exp_data.zip",
        "columns_schema": ["title", "authors", "venue", "year"],
    },
    # Dirty/DBLP-ACM: same underlying data as Structured/DBLP-ACM but with
    # attribute-value swaps and NULL injection (the ER-Magellan "dirty"
    # variant). Ditto ships this in its repo; no external download needed.
    # Added to support the §V.F.5 robustness-under-dirty-data extension.
    "Dirty/DBLP-ACM": {
        "type": "Dirty",
        "short": "DA-Dirty",
        "ditto_task": "Dirty/DBLP-ACM",
        "ditto_subdir": "data/er_magellan/Dirty/DBLP-ACM",
        "uw_url": None,
        "zip_url": None,
        "columns_schema": ["title", "authors", "venue", "year"],
    },
    "Textual/Abt-Buy": {
        "type": "Textual",
        "short": "AB",
        "ditto_task": "Textual/Abt-Buy",
        "ditto_subdir": "data/er_magellan/Textual/Abt-Buy",
        "uw_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/",
        "zip_url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/abt_buy_exp_data.zip",
        "columns_schema": ["name", "description", "price"],
    },
    # Textual/Company: companies dataset from Magellan/deepmatcher.
    # Not bundled with Ditto, and UW ships no zip — only individual CSVs
    # (tableA 185M, tableB 96M; download.py uses files-mode via uw_url).
    # Single-attribute schema ("content" — long-form Wikipedia-style company text).
    # Included for schema diversity in the L3 tier.
    "Textual/Company": {
        "type": "Textual",
        "short": "CO",
        "ditto_task": "Textual/Company",
        "ditto_subdir": "data/er_magellan/Textual/Company",
        "uw_url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/exp_data/",
        "zip_url": None,
        "columns_schema": ["content"],
    },
}

# Pilot: one dataset end-to-end before scaling up.
PILOT_DATASET = "Structured/Walmart-Amazon"


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# For Experiment (label efficiency). "full" = use entire train set.
K_VALUES: list = [50, 100, 200, 500, "full"]

# For Experiment (the 5-50 cross-domain gap, the novelty contribution).
K_VALUES_LOW: list = [5, 10, 25, 50]

SEEDS: list[int] = [42, 123, 456]


# ---------------------------------------------------------------------------
# Ditto hyperparameters (mirror the paper / repo defaults)
# ---------------------------------------------------------------------------

DITTO_HPARAMS = {
    "lm": "roberta",           # paper best; switch to "distilbert" for Colab speed
    "batch_size": 32,
    "max_len": 256,
    "lr": 3e-5,
    "n_epochs": 15,
    "fp16": False,
    "da": "del",               # data augmentation operator
    "dk": None,                # domain knowledge (set per-dataset if desired)
    "summarize": False,
}

# Per-dataset overrides (Ditto's configs.json already sets these, but we
# carry them here so our runs are reproducible without relying on that file).
DITTO_HPARAMS_BY_DATASET = {
    "Structured/Walmart-Amazon": {"dk": None, "max_len": 256},
    "Structured/Amazon-Google":  {"dk": "product", "max_len": 256},
    "Structured/DBLP-ACM":       {"dk": None,      "max_len": 180},
    "Dirty/DBLP-ACM":            {"dk": None,      "max_len": 180},   # same schema as Structured/DBLP-ACM
    "Textual/Abt-Buy":           {"dk": "product", "max_len": 256},
    # WDC product corpora
    # Source data ships in ditto/data/wdc/<cat>/ (small/medium/large/xlarge);
    # our pipeline copies the .small variant into data/processed/wdc/<cat>/.
    "wdc/computers":             {"dk": "product", "max_len": 256},
    "wdc/cameras":               {"dk": "product", "max_len": 256},
    "wdc/watches":               {"dk": "product", "max_len": 256},
    "wdc/shoes":                 {"dk": "product", "max_len": 256},
}


def get_ditto_hparams(dataset_key: str) -> dict:
    """Return merged base + per-dataset hparams."""
    h = dict(DITTO_HPARAMS)
    h.update(DITTO_HPARAMS_BY_DATASET.get(dataset_key, {}))
    return h


def ensure_dirs() -> None:
    """Create all configured dirs. Idempotent."""
    for p in [
        PATHS.data_raw, PATHS.data_processed, PATHS.data_splits,
        PATHS.models, PATHS.results_runs, PATHS.results_figures,
        PATHS.results_tables,
    ]:
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"BASE: {BASE}")
    print(f"Exists: {BASE.exists()}")
    ensure_dirs()
    for field in PATHS.__dataclass_fields__:
        print(f"  {field:18s} -> {getattr(PATHS, field)}")
    print(f"\nDatasets: {list(ERM_DATASETS.keys())}")
    print(f"Pilot:    {PILOT_DATASET}")
    print(f"K values: {K_VALUES}")
    print(f"Seeds:    {SEEDS}")
