"""
Create K-shot training splits for experiment (label efficiency).

Operates on Ditto-format .txt files from data/processed/. Each K-shot
split gets its own directory with train.txt, valid.txt, test.txt, where
valid and test are unchanged from the full dataset and only train is
subsampled.

Sampling is stratified: we preserve the positive:negative ratio of the
full train set, with at least 1 positive and 1 negative in every split
(so K=5 still has a match to learn from).

Usage:
    python -m src.data.make_splits                              # pilot + all K + all seeds
    python -m src.data.make_splits Structured/Walmart-Amazon    # one dataset
    python -m src.data.make_splits --k 50 --seed 42 Structured/Walmart-Amazon
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from config.config import (
    ERM_DATASETS,
    K_VALUES,
    K_VALUES_LOW,
    PATHS,
    PILOT_DATASET,
    SEEDS,
    ensure_dirs,
)


def _read_ditto_txt(path: Path) -> list[tuple[str, int]]:
    """Return list of (raw_line_without_label, label_int)."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit("\t", 1)  # label is the last tab-separated field
        if len(parts) != 2:
            continue
        body, label = parts
        try:
            pairs.append((body, int(label)))
        except ValueError:
            continue
    return pairs


def _stratified_k_shot(
    pairs: list[tuple[str, int]],
    k: int,
    seed: int,
) -> list[tuple[str, int]]:
    """
    Sample k pairs preserving the positive ratio. Guarantees >=1 pos and
    >=1 neg when both exist in the pool.
    """
    rng = random.Random(seed)
    pos = [p for p in pairs if p[1] == 1]
    neg = [p for p in pairs if p[1] == 0]

    if not pos or not neg:
        # Degenerate dataset; just sample uniformly
        return rng.sample(pairs, k=min(k, len(pairs)))

    ratio = len(pos) / len(pairs)
    k_pos = max(1, round(k * ratio))
    k_neg = max(1, k - k_pos)

    # Clamp to available
    k_pos = min(k_pos, len(pos))
    k_neg = min(k_neg, len(neg))

    sampled = rng.sample(pos, k_pos) + rng.sample(neg, k_neg)
    rng.shuffle(sampled)
    return sampled


def _write_ditto_txt(path: Path, pairs: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{body}\t{label}" for body, label in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_split(
    dataset_key: str,
    k: int | str,
    seed: int,
    *,
    force: bool = False,
) -> Path:
    """
    Build one (dataset, K, seed) split.

    Args:
        k: integer label count, or "full" to copy the entire train set.

    Returns:
        Path to the split directory.
    """
    processed_dir = PATHS.data_processed / dataset_key
    if not (processed_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"Processed data missing: {processed_dir}. "
            f"Run `python -m src.data.convert_to_ditto {dataset_key}` first."
        )

    tag = f"k_{k}_seed_{seed}"
    split_dir = PATHS.data_splits / dataset_key / tag

    if (split_dir / "train.txt").exists() and not force:
        print(f"  [{dataset_key}] {tag} already exists (skip)")
        return split_dir

    # Load and subsample train only; valid/test stay full
    full_train = _read_ditto_txt(processed_dir / "train.txt")

    if k == "full":
        train_sample = full_train
        actual_k = len(full_train)
    else:
        train_sample = _stratified_k_shot(full_train, int(k), seed)
        actual_k = len(train_sample)

    _write_ditto_txt(split_dir / "train.txt", train_sample)

    # Copy valid/test unchanged (read+write to keep everything as ditto .txt)
    for split in ("valid", "test"):
        pairs = _read_ditto_txt(processed_dir / f"{split}.txt")
        _write_ditto_txt(split_dir / f"{split}.txt", pairs)

    n_pos = sum(1 for _, lbl in train_sample if lbl == 1)
    n_neg = actual_k - n_pos
    print(f"  [{dataset_key}] {tag}: {actual_k} train ({n_pos} pos, {n_neg} neg)")
    return split_dir


def make_all_splits(
    datasets: list[str] | None = None,
    k_values: list | None = None,
    seeds: list[int] | None = None,
    *,
    force: bool = False,
) -> None:
    datasets = datasets or [PILOT_DATASET]  # default: pilot only
    k_values = k_values or K_VALUES
    seeds = seeds or SEEDS

    ensure_dirs()
    print(f"datasets: {datasets}")
    print(f"K values: {k_values}")
    print(f"seeds:    {seeds}")

    for ds in datasets:
        print(f"\n=== {ds} ===")
        for k in k_values:
            for seed in seeds:
                try:
                    make_split(ds, k, seed, force=force)
                except Exception as e:
                    print(f"  FAILED (K={k}, seed={seed}): {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*",
                        help="Dataset keys (default: pilot only)")
    parser.add_argument("--k", type=str, default=None,
                        help="Single K value (default: all in K_VALUES)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: all in SEEDS)")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run over every dataset in ERM_DATASETS")
    parser.add_argument("--low-k", action="store_true",
                        help="Use K_VALUES_LOW (5,10,25,50) for Experiment 3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.all_datasets:
        datasets = list(ERM_DATASETS.keys())
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = [PILOT_DATASET]

    if args.k is not None:
        k_values: list = [int(args.k) if args.k.isdigit() else args.k]
    elif args.low_k:
        k_values = K_VALUES_LOW
    else:
        k_values = K_VALUES

    seeds = [args.seed] if args.seed is not None else SEEDS

    make_all_splits(datasets, k_values, seeds, force=args.force)


if __name__ == "__main__":
    main()
