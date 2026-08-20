"""
Parse Ditto's training stdout to extract final F1/precision/recall.

Ditto's train_ditto.py prints per-epoch lines like:
    epoch 5: dev_f1=0.9512 f1=0.9487 best_dev_f1=0.9512
and a final block with the best dev/test scores. We look for the last
matching line to get the final reported F1.

We also support parsing the predictions file produced by matcher.py if
you prefer to recompute metrics from raw predictions (more trustworthy).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


EPOCH_LINE_RE = re.compile(
    r"epoch\s+(\d+)[:\s].*?dev_f1\s*=\s*([0-9.]+).*?f1\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)
BEST_DEV_RE = re.compile(r"best_dev_f1\s*=\s*([0-9.]+)", re.IGNORECASE)


def parse_training_log(text: str) -> dict | None:
    """
    Return the metrics from the epoch with the highest dev_f1.

    Ditto reports test f1 on every epoch. We pick the epoch whose dev_f1
    is the best, and report that epoch's test f1. This matches how the
    paper reports numbers (best dev selected, test reported).
    """
    best = None  # (dev_f1, epoch, test_f1)
    for m in EPOCH_LINE_RE.finditer(text):
        epoch = int(m.group(1))
        dev_f1 = float(m.group(2))
        test_f1 = float(m.group(3))
        if best is None or dev_f1 > best[0]:
            best = (dev_f1, epoch, test_f1)

    if best is None:
        return None

    return {
        "best_epoch": best[1],
        "dev_f1": best[0],
        "test_f1": best[2],
    }


def parse_predictions_jsonl(
    pred_path: Path,
    test_txt_path: Path,
) -> dict:
    """
    Recompute F1/P/R from matcher.py's jsonl predictions against the
    ground-truth labels in the test .txt.

    Safer than parsing logs when the log format changes.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # matcher.py uses 'match' for the predicted label (0/1)
            preds.append(int(obj["match"]))

    labels = []
    for line in test_txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit("\t", 1)
        if len(parts) == 2:
            labels.append(int(parts[1]))

    if len(preds) != len(labels):
        raise ValueError(
            f"length mismatch: {len(preds)} preds vs {len(labels)} labels"
        )

    return {
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "n_preds": len(preds),
        "n_positive_preds": sum(preds),
        "n_positive_labels": sum(labels),
    }
