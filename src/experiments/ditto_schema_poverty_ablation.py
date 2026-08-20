"""
Exp 16 Ditto ablation — SFT paradigm column of the schema-poverty ladder.

Companion to `notebooks/05_schema_poverty_ablation_llm.ipynb` (LLM column via API).
This driver runs Ditto warm-start K=100 at each (rung, perturbation) cell on the
target pairs, using the same field-masking and contamination-perturbation logic
as the LLM notebook so the two paradigms are compared like-for-like.

    python -m src.experiments.ditto_schema_poverty_ablation \
        --source Structured/Walmart-Amazon \
        --target Structured/Amazon-Google \
        --rung 4 --perturbation none --seed 42 --k 100

Output:
    results/runs/sparsity-ablation-sft/
        {source}__{target}/rung_{r}/perturb_{p}/seed_{s}/metrics.json

Prerequisites:
    - Source K=full seed=42 checkpoint from Exp 1 (path resolved via `_find_source_pt`)
    - Ditto AdamW patch applied and __pycache__ cleared (see docs/methods/ditto_setup.md)
    - field_discriminativeness.csv present under results/schema_poverty_ablation/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import PATHS, ensure_dirs, get_ditto_hparams
from src.experiments.ditto_warmstart import _patch_ditto_configs, _task_name_for_run
from src.experiments.ditto_warmstart import _find_source_pt
from src.utils.ditto_parser import parse_training_log


METHOD_ABLATION = "sparsity-ablation-sft"

# Match one COL <name> VAL <value> segment.
FIELD_RE = re.compile(r"COL\s+(\S+)\s+VAL\s*(.*?)\s*(?=COL\s+|$)")


# ---------------------------------------------------------------------------
# Field-set derivation from the MI-ranked drop ladder (mirrors schema-poverty ablation notebook)
# ---------------------------------------------------------------------------

def load_mi_ranking(csv_path: Path | None = None) -> pd.DataFrame:
    """Return field_discriminativeness rows keyed by (dataset, ordered ascending by MI)."""
    csv_path = csv_path or (PATHS.base / "results" / "schema_poverty_ablation" / "field_discriminativeness.csv")
    return pd.read_csv(csv_path)


def rung_field_set(target: str, rung: int, mi_df: pd.DataFrame) -> tuple[set[str], bool]:
    """
    Return (retained_fields, truncate_top_field_to_3_tokens) for the given rung.

    Ladder semantics — must match the LLM notebook's cell 9 exactly so the two
    paradigms are compared on identical inputs:

        rung 0  full schema (all attributes)
        rung 1  drop the single lowest-MI attribute
        rung 2  drop the two lowest-MI attributes
        rung 3  keep only the top-MI attribute (no truncation)
        rung 4  keep only the top-MI attribute AND truncate to three tokens

    For 3-field targets rung 2 == rung 3 (both {top field}); the driver's
    rung-dedup guard skips the duplicate cell.
    """
    fields_asc = mi_df[mi_df["dataset"] == target].sort_values("mi")["field"].tolist()
    fields_desc = list(reversed(fields_asc))
    # Targets not present in the MI CSV (e.g. wdc/watches, called from transitivity-test runner)
    # can still request rung 0 (full schema, no masking). We return None as a
    # sentinel for "no masking"; mask_record handles it by passing the record
    # through unchanged.
    if not fields_asc:
        if rung == 0:
            return None, False
        raise ValueError(
            f"target {target!r} not found in MI ranking CSV; add it via "
            f"`python -m src.analysis.field_discriminativeness --datasets {target}` "
            f"before running any rung > 0 ablation cell."
        )
    if rung == 0:
        return set(fields_asc), False
    if rung == 1:
        return set(fields_asc[1:]), False
    if rung == 2:
        return set(fields_asc[2:]), False
    if rung == 3:
        return {fields_desc[0]}, False
    if rung == 4:
        return {fields_desc[0]}, True
    raise ValueError(f"unknown rung: {rung}")


# ---------------------------------------------------------------------------
# Record parsing, masking, contamination perturbations (mirrors schema-poverty ablation notebook)
# ---------------------------------------------------------------------------

_ABBREV = {
    "corporation": "corp", "company": "co", "limited": "ltd", "incorporated": "inc",
    "international": "intl", "brothers": "bros", "systems": "sys",
    "street": "st", "avenue": "ave", "road": "rd", "boulevard": "blvd",
    "united states": "us", "university": "univ",
}


def parse_record(rec_str: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in FIELD_RE.finditer(rec_str)]


def format_record(fields: list[tuple[str, str]]) -> str:
    return " ".join(f"COL {n} VAL {v}" for n, v in fields)


def mask_record(rec_str: str, keep_fields: set[str] | None, truncate_title: bool) -> str:
    # keep_fields = None is the sentinel for "no masking, pass through unchanged"
    # (used when rung 0 is requested for a target not in the MI CSV; see
    # rung_field_set).
    if keep_fields is None:
        return rec_str
    fields = parse_record(rec_str)
    kept = [(n, v) for n, v in fields if n in keep_fields]
    if truncate_title:
        # At the truncated stub rung, exactly one field is retained (the top-MI
        # field). Truncate whichever field it is — the previous 'n == "title"'
        # guard was a latent bug for targets whose top-MI field is not literally
        # named 'title' (Abt-Buy's is 'name'; Walmart-Amazon's is 'modelno').
        kept = [(n, " ".join(v.split()[:3])) for n, v in kept]
    return format_record(kept)


def _short_form(v: str) -> str:
    return " ".join(_ABBREV.get(t.lower(), t) for t in v.split())


def _glued(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", v).upper()


def _initials(v: str) -> str:
    toks = v.split()
    return ".".join(t[0].upper() for t in toks if t) + "." if len(toks) >= 2 else v


def _shuffle(v: str, seed: int) -> str:
    r = np.random.default_rng(seed)
    toks = v.split()
    perm = r.permutation(len(toks))
    return " ".join(toks[i] for i in perm)


def perturb_record(rec_str: str, perturbation: str, seed: int) -> str:
    if perturbation == "none":
        return rec_str
    fn_map = {
        "short_form": _short_form,
        "glued": _glued,
        "initials": _initials,
        "shuffle": lambda v: _shuffle(v, seed),
    }
    fn = fn_map[perturbation]
    fields = [(n, fn(v) if v else v) for n, v in parse_record(rec_str)]
    return format_record(fields)


def transform_line(line: str, keep_fields: set[str], truncate: bool,
                    perturbation: str, seed_left: int, seed_right: int) -> str | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        return None
    left, right, label = parts
    left = mask_record(left, keep_fields, truncate)
    right = mask_record(right, keep_fields, truncate)
    left = perturb_record(left, perturbation, seed_left)
    right = perturb_record(right, perturbation, seed_right)
    return f"{left}\t{right}\t{label}\n"


# ---------------------------------------------------------------------------
# Ditto data-tree write + config patch
# ---------------------------------------------------------------------------

def _task_dir(source: str, target: str, rung: int, perturbation: str, seed: int) -> str:
    return (f"{target.replace('/', '__')}__abl__rung{rung}"
            f"__{perturbation}__s{seed}__from__{source.replace('/', '__')}")


def _write_ablation_task(
    source: str, target: str, rung: int, perturbation: str, seed: int,
    k: int, mi_df: pd.DataFrame,
) -> str:
    """
    Materialise the masked+perturbed K-sampled train + full valid + full test into
    Ditto's data tree under a fresh task subdirectory, then patch configs.json.

    Returns the Ditto task name.
    """
    keep, truncate = rung_field_set(target, rung, mi_df)

    src_root = PATHS.data_processed / target
    if not (src_root / "train.txt").exists():
        raise FileNotFoundError(f"expected {src_root}/train.txt to exist — run splits first")

    task_subdir = _task_dir(source, target, rung, perturbation, seed)
    ditto_task_root = PATHS.ditto_repo / "data" / task_subdir
    ditto_task_root.mkdir(parents=True, exist_ok=True)

    # Sample K rows from source train (seed-fixed, class-stratified would be nicer;
    # for simplicity we use uniform random consistent with the from-scratch convention).
    train_lines = (src_root / "train.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    if k != "full" and k < len(train_lines):
        r = np.random.default_rng(seed)
        idx = sorted(r.choice(len(train_lines), size=k, replace=False))
        train_lines = [train_lines[i] for i in idx]

    for split_name, split_lines in [
        ("train.txt", train_lines),
        ("valid.txt", (src_root / "valid.txt").read_text(encoding="utf-8").splitlines(keepends=True)),
        ("test.txt", (src_root / "test.txt").read_text(encoding="utf-8").splitlines(keepends=True)),
    ]:
        out_lines = []
        for i, line in enumerate(split_lines):
            transformed = transform_line(
                line, keep, truncate, perturbation,
                seed_left=seed, seed_right=seed + 1,  # per-record seed offset for shuffle
            )
            if transformed:
                out_lines.append(transformed)
        (ditto_task_root / split_name).write_text("".join(out_lines), encoding="utf-8")

    task_name = task_subdir
    _register_task_in_ditto_configs(task_name, target, task_subdir)
    return task_name


def _register_task_in_ditto_configs(task_name: str, target: str, task_subdir: str) -> None:
    """Append the ablation task to Ditto's configs.json if missing.

    If the exact target is not present in configs.json (e.g. wdc/watches called
    from transitivity test when Ditto's shipped configs only include a subset of WDC pairs),
    fall back to using any existing entry as a template — classification EM
    tasks share task_type and vocab; only the paths differ.
    """
    cfg_path = PATHS.ditto_repo / "configs.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if any(entry["name"] == task_name for entry in cfg):
        return
    template = None
    for entry in cfg:
        if entry["name"] == target:
            template = entry
            break
    if template is None:
        if not cfg:
            raise ValueError(f"Ditto configs.json is empty — cannot register {task_name!r}")
        template = cfg[0]
        print(f"  [warn] template task {target!r} not found in Ditto configs.json; "
              f"falling back to {template['name']!r} as template for task_type/vocab. "
              f"Add a proper {target!r} entry to configs.json for future runs.")
    new = dict(template)
    new["name"] = task_name
    new["trainset"] = f"data/{task_subdir}/train.txt"
    new["validset"] = f"data/{task_subdir}/valid.txt"
    new["testset"] = f"data/{task_subdir}/test.txt"
    cfg.append(new)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_ablation_cell(
    source: str, target: str, rung: int, perturbation: str,
    seed: int, k: int = 100, *, skip_if_done: bool = True,
    source_pt_override: Path | str | None = None,
) -> dict | None:
    ensure_dirs()

    run_dir = (PATHS.results_runs / METHOD_ABLATION
                / f"{source.replace('/', '__')}__{target.replace('/', '__')}"
                / f"rung_{rung}" / f"perturb_{perturbation}" / f"seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    if skip_if_done and metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        if m.get("returncode") == 0 and m.get("test_f1") is not None:
            print(f"  [skip] {source}->{target} rung={rung} p={perturbation} s={seed} — "
                  f"already done: test_f1={m['test_f1']:.4f}")
            return m

    if source_pt_override is not None:
        source_pt = Path(source_pt_override)
        if not source_pt.exists():
            raise FileNotFoundError(f"source_pt_override does not exist: {source_pt}")
    else:
        source_pt = _find_source_pt(source)
    mi_df = load_mi_ranking()

    print(f"\n[ablation] {source} -> {target}  rung={rung}  perturb={perturbation}  seed={seed}")
    keep, truncate = rung_field_set(target, rung, mi_df)
    keep_repr = "(all fields, no masking)" if keep is None else sorted(keep)
    print(f"  retained fields: {keep_repr}  truncate={truncate}")

    # Rung-dedup guard: skip a rung whose (field-set, truncate) is identical to
    # the previous rung's — matches the LLM notebook's dedup guard so 3-field
    # targets (Abt-Buy, Amazon-Google) skip rung 3, whose retained {top field}
    # equals rung 2 for them.
    if rung > 0 and keep is not None:
        prev_keep, prev_trunc = rung_field_set(target, rung - 1, mi_df)
        if prev_keep == keep and prev_trunc == truncate:
            print(f"  [rung dedup] rung {rung} identical to rung {rung-1} for {target}; skipping.")
            return None

    task_name = _write_ablation_task(source, target, rung, perturbation, seed, k, mi_df)
    print(f"  Ditto task: {task_name}")

    hparams = get_ditto_hparams(target)
    cmd = [
        "python", "train_ditto.py",
        "--task", task_name,
        "--batch_size", str(hparams["batch_size"]),
        "--max_len", str(hparams["max_len"]),
        "--lr", str(hparams["lr"]),
        "--n_epochs", str(hparams["n_epochs"]),
        "--lm", hparams["lm"],
        "--run_id", str(seed),
        "--logdir", str(PATHS.models / task_name),
        "--save_model",
        "--checkpoint_path", str(source_pt),
    ]

    log_path = run_dir / "train.log"
    start = time.time()
    with open(log_path, "w") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n")
        log_f.write(f"# cwd: {PATHS.ditto_repo}\n\n")
        proc = subprocess.run(cmd, cwd=PATHS.ditto_repo, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    parsed = parse_training_log(log_path.read_text()) if log_path.exists() else None
    metrics = {
        "source_dataset": source,
        "target_dataset": target,
        "rung": rung,
        "perturbation": perturbation,
        "seed": seed,
        "k": k,
        "retained_fields": sorted(keep) if keep is not None else "(all fields, no masking)",
        "truncate_title": truncate,
        "task_name": task_name,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        **(parsed or {}),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  test_f1 = {metrics.get('test_f1')}  ({elapsed:.0f}s, rc={proc.returncode})")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rung", type=int, required=True)
    parser.add_argument("--perturbation", default="none",
                        choices=["none", "short_form", "glued", "initials", "shuffle"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()
    run_ablation_cell(
        args.source, args.target, args.rung, args.perturbation, args.seed,
        k=args.k, skip_if_done=not args.no_skip,
    )


if __name__ == "__main__":
    main()
