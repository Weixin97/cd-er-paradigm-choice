"""
Ditto warm-start helpers.

Path resolvers for locating the source-trained checkpoint (`.pt`) that
`ditto_schema_poverty_ablation.py` and the results-aggregation code depend on.

The full Ditto warm-start training loop (loading the source checkpoint,
fine-tuning on target K-shot, evaluating on the target test set) is
straightforward to reproduce from Ditto's own repository
(megagonlabs/ditto) by pointing `--checkpoint_path` at the checkpoint that
`_find_source_pt()` returns. See the notebook `notebooks/README.md` for
detailed reproduction instructions.

Per-cell measurements are already provided in
`results/runs/ditto-warmstart/<source>__from__<target>/k_100_seed_<s>/`.
"""
from __future__ import annotations

import json
from pathlib import Path

from config.config import PATHS


METHOD_WARMSTART = "ditto-warmstart"
METHOD_ZEROSHOT = "ditto-zeroshot"
METHOD_FROMSCRATCH = "ditto"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_pair(source_key: str, target_key: str) -> str:
    """Filesystem-safe encoding of a (source, target) pair."""
    return f"{source_key.replace('/', '__')}__from__{target_key.replace('/', '__')}"


def _warmstart_run_dir(source_key: str, target_key: str,
                        k: int | str, seed: int) -> Path:
    """Directory for a single warm-start run's metrics.json + train.log."""
    return (PATHS.results_runs / METHOD_WARMSTART /
            _safe_pair(source_key, target_key) / f"k_{k}_seed_{seed}")


def _zeroshot_run_dir(source_key: str, target_key: str) -> Path:
    """Directory for a single zero-shot run's metrics.json."""
    return PATHS.results_runs / METHOD_ZEROSHOT / _safe_pair(source_key, target_key)


def _find_source_pt(source_key: str) -> Path:
    """
    Locate the source-trained model.pt for warm-starting.

    Convention: source models are trained on the full source dataset
    (K = full, seed = 42) and stored under
      PATHS.models / "<source_key>-kfull-s42" / "<source_key>-kfull-s42" / model.pt

    Falls back to any *.pt under the source directory in case the naming
    convention changes.
    """
    base = PATHS.models / f"{source_key}-kfull-s42"
    pts = sorted(base.rglob("model.pt")) if base.exists() else []
    if not pts and base.exists():
        pts = sorted(base.rglob("*.pt"))
    if not pts:
        raise FileNotFoundError(
            f"No .pt file found under {base}. "
            f"Train the source model first with Ditto's standard training script "
            f"(megagonlabs/ditto) using --task {source_key} --n_epochs 15 --batch_size 32 "
            f"and store the checkpoint at {base}/model.pt."
        )
    return pts[0]


# ---------------------------------------------------------------------------
# Ditto task-config helpers (used by the schema-poverty ablation runner)
# ---------------------------------------------------------------------------

def _task_name_for_run(dataset_key: str, k: int | str, seed: int) -> str:
    """
    Build a unique Ditto task name for a (dataset, K, seed) combination.
    Ditto's configs.json uses slashes, so we keep the same structure:
        Structured/Walmart-Amazon -> Structured/Walmart-Amazon-k100-s42
    """
    return f"{dataset_key}-k{k}-s{seed}"


def _patch_ditto_configs(dataset_key: str, k: int | str, seed: int) -> None:
    """
    Add an entry to ditto/configs.json for our custom task name, pointing
    at the symlinked split data. Idempotent.
    """
    configs_path = PATHS.ditto_repo / "configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(f"Ditto configs.json not found at {configs_path}")

    task_name = _task_name_for_run(dataset_key, k, seed)
    with open(configs_path) as f:
        configs = json.load(f)

    # Find the base config for this dataset and clone it under our task name
    base_cfg = next((c for c in configs if c.get("name") == dataset_key), None)
    if base_cfg is None and dataset_key.startswith("wdc/"):
        # WDC keys in our pipeline use "wdc/<cat>"; Ditto names them "wdc_<cat>_small".
        cat = dataset_key.split("/", 1)[1]
        ditto_name = f"wdc_{cat}_small"
        base_cfg = next((c for c in configs if c.get("name") == ditto_name), None)
    if base_cfg is None:
        raise ValueError(
            f"Ditto has no base config for '{dataset_key}'. "
            f"Available: {[c['name'] for c in configs[:10]]}..."
        )

    # Build new config entry. Paths are relative to Ditto repo root.
    task_rel = f"data/er_magellan/{task_name}"
    new_cfg = dict(base_cfg)
    new_cfg["name"] = task_name
    new_cfg["trainset"] = f"{task_rel}/train.txt"
    new_cfg["validset"] = f"{task_rel}/valid.txt"
    new_cfg["testset"]  = f"{task_rel}/test.txt"

    # Upsert
    configs = [c for c in configs if c.get("name") != task_name]
    configs.append(new_cfg)

    with open(configs_path, "w") as f:
        json.dump(configs, f, indent=2)
