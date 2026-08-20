"""
SFT paradigm aggregation for Table... 

Walks results/runs/ across six method families with heterogeneous
metrics.json schemas, normalises source/target naming, assigns 
pair IDs (P1..P9), and emits:

  results/analysis/paradigm_all.csv         one row per metrics.json
  results/analysis/table8_sft_aggregate.csv 4.3
  results/analysis/table10_cross_paradigm.csv  4.4
  results/analysis/paper_snippets.md        Markdown table strings ready to paste

Method families handled:

  ditto-zeroshot         source-trained checkpoint, no target labels
  ditto-warmstart        source-trained + K target-label fine-tune
  ditto-vanilla-baseline from-scratch on K target labels (no source)
  ditto                  earlier variant, same schema as vanilla-baseline
  dader-mmd, dader-mmd-ep15 domain-adversarial alignment, zero target
  matchgpt_k10      LLM prompting, in-domain demonstrations

Usage inside the analysis Docker container:

    docker compose exec analysis bash -c \
        "cd $REPO_ROOT && python -m src.analysis.sft_results_aggregation"
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config.config import PATHS


# ---------------------------------------------------------------------------
# Canonical naming
# ---------------------------------------------------------------------------

# Short-code → canonical dataset name (DADER uses shortcodes in its metrics)
DADER_ALIAS = {
    "wa1": "Structured/Walmart-Amazon",
    "wa": "Structured/Walmart-Amazon",
    "ab": "Textual/Abt-Buy",
    "ag": "Structured/Amazon-Google",
    "da": "Structured/DBLP-ACM",
    "ddaa": "Dirty/DBLP-ACM",
    "ds": "Structured/DBLP-Scholar",
    "ia": "Structured/iTunes-Amazon",
    "computers": "wdc/computers",
    "watches": "wdc/watches",
    "cameras": "wdc/cameras",
    "shoes": "wdc/shoes",
}

# (source_canonical, target_canonical) → main paper pair ID.
# Source is None for from-scratch / in-domain / LLM in-domain few-shot.
PAIRS = {
    ("Structured/Walmart-Amazon", "Textual/Abt-Buy"): "P1",
    ("wdc/computers", "wdc/watches"): "P2",
    ("Structured/Walmart-Amazon", "Structured/DBLP-ACM"): "P3",
    ("Structured/Walmart-Amazon", "Structured/Amazon-Google"): "P5",
    ("Structured/Walmart-Amazon", "Dirty/DBLP-ACM"): "P6",
    ("Structured/iTunes-Amazon", "Structured/DBLP-ACM"): "P7",
    ("Structured/iTunes-Amazon", "Structured/DBLP-Scholar"): "P8",
    ("Structured/Walmart-Amazon", "Structured/DBLP-Scholar"): "P9",
}

# For rows with no source (from-scratch / in-domain), map the target alone
# to a pair suffix so downstream analysis can compare against warm-start.
TARGET_ONLY_PAIRS = {
    "Textual/Abt-Buy": "P1",
    "wdc/watches": "P2",
    "Structured/DBLP-ACM": "P3",
    "Structured/Amazon-Google": "P5",
    "Dirty/DBLP-ACM": "P6",
    "Structured/DBLP-Scholar": "P9",
}


def _canonical_dataset(name: str | None) -> str | None:
    """Normalise a dataset name from any schema to the canonical form."""
    if name is None or name == "":
        return None
    if name in DADER_ALIAS:
        return DADER_ALIAS[name]
    if "__" in name and "/" not in name:
        return name.replace("__", "/", 1)
    return name


# ---------------------------------------------------------------------------
# Per-family parsers
#
# Each parser receives one metrics.json dict and returns a normalised row
# dict, or None if the record should be skipped.
# ---------------------------------------------------------------------------

def _paradigm_for(method: str) -> str:
    if method.startswith("ditto") or method.startswith("dader"):
        return "sft"
    if method.startswith("matchgpt") or method.startswith("cider"):
        return "llm"
    return "unknown"


def _parse_ditto_warmstart(d: dict) -> dict:
    return {
        "method": d.get("method", "ditto-warmstart"),
        "paradigm": "sft",
        "config": "warm-start",
        "source": _canonical_dataset(d.get("source")),
        "target": _canonical_dataset(d.get("target")),
        "k": d.get("k"),
        "seed": d.get("seed"),
        "test_f1": d.get("test_f1"),
        "test_precision": d.get("test_precision"),
        "test_recall": d.get("test_recall"),
        "elapsed_sec": d.get("elapsed_sec"),
    }


def _parse_ditto_zeroshot(d: dict) -> dict:
    return {
        "method": "ditto-zeroshot",
        "paradigm": "sft",
        "config": "zero-shot",
        "source": _canonical_dataset(d.get("source")),
        "target": _canonical_dataset(d.get("target")),
        "k": 0,
        "seed": None,
        "test_f1": d.get("test_f1"),
        "test_precision": d.get("test_precision"),
        "test_recall": d.get("test_recall"),
        "elapsed_sec": d.get("elapsed_sec"),
    }


def _parse_ditto_scratch(d: dict) -> dict:
    """ditto-vanilla-baseline and legacy 'ditto' both have `dataset` and no source."""
    return {
        "method": d.get("method", "ditto"),
        "paradigm": "sft",
        "config": "from-scratch",
        "source": None,
        "target": _canonical_dataset(d.get("dataset")),
        "k": d.get("k"),
        "seed": d.get("seed"),
        "test_f1": d.get("test_f1"),
        "test_precision": d.get("test_precision"),
        "test_recall": d.get("test_recall"),
        "elapsed_sec": d.get("elapsed_sec"),
    }


def _parse_dader(d: dict) -> dict:
    return {
        "method": d.get("method", "dader"),
        "paradigm": "sft",
        "config": "adversarial-zero-target",
        "source": _canonical_dataset(d.get("src")),
        "target": _canonical_dataset(d.get("tgt")),
        "k": 0,
        "seed": None,
        "test_f1": d.get("test_f1"),
        "test_precision": d.get("test_precision"),
        "test_recall": d.get("test_recall"),
        "elapsed_sec": d.get("elapsed_sec"),
    }


def _parse_matchgpt(d: dict) -> dict:
    # Two schema variants in the wild:
    #   in-domain runs (K=10 random, zero_shot): "dataset" field only
    #   cross-domain runs (K=2 random, K=10 related): "source_dataset" + "target_dataset"
    target = d.get("target_dataset") or d.get("dataset")
    return {
        "method": "matchgpt_k10",
        "paradigm": "llm",
        "config": d.get("setting", ""),
        "source": _canonical_dataset(d.get("source_dataset")),
        "target": _canonical_dataset(target),
        "k": d.get("k_demos"),
        "seed": d.get("seed"),   # deterministic if absent
        "backbone": d.get("model"),
        "demo_strategy": d.get("demo_strategy"),
        "test_f1": d.get("test_f1"),
        "test_precision": d.get("test_precision"),
        "test_recall": d.get("test_recall"),
        "elapsed_sec": d.get("elapsed_sec"),
    }


PARSER_BY_METHOD = {
    "ditto-warmstart": _parse_ditto_warmstart,
    "ditto-zeroshot": _parse_ditto_zeroshot,
    "ditto-vanilla-baseline": _parse_ditto_scratch,
    "ditto": _parse_ditto_scratch,
    "dader-mmd": _parse_dader,
    "dader-mmd-ep15": _parse_dader,
    "dader": _parse_dader,
    "matchgpt_k10": _parse_matchgpt,
}


def _pair_id(source: str | None, target: str | None) -> str | None:
    if target is None:
        return None
    if source is not None and (source, target) in PAIRS:
        return PAIRS[(source, target)]
    if source is None and target in TARGET_ONLY_PAIRS:
        return TARGET_ONLY_PAIRS[target]
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def collect(runs_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_file in runs_root.rglob("metrics.json"):
        try:
            data = json.loads(metrics_file.read_text())
        except Exception as e:
            print(f"  skip (bad json): {metrics_file} — {e}")
            continue

        method_key = data.get("method", "")
        parser = PARSER_BY_METHOD.get(method_key)
        if parser is None:
            # Try prefix match — some methods have suffixes
            for prefix, fn in PARSER_BY_METHOD.items():
                if method_key.startswith(prefix):
                    parser = fn
                    break
        if parser is None:
            print(f"  skip (unknown method '{method_key}'): {metrics_file}")
            continue

        row = parser(data)
        row["pair_id"] = _pair_id(row.get("source"), row.get("target"))
        row["source_file"] = str(metrics_file.relative_to(runs_root.parent))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    front = [
        "pair_id", "paradigm", "method", "config",
        "source", "target", "k", "seed",
        "test_f1", "test_precision", "test_recall",
        "elapsed_sec", "source_file",
    ]
    others = [c for c in df.columns if c not in front]
    return df[[c for c in front if c in df.columns] + others]


# ---------------------------------------------------------------------------
# main paper table builders
# ---------------------------------------------------------------------------

def build_ditto_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """main paper Sec. 4.3 Table 8: per-pair SFT paradigm F1.

    Columns: pair | Ditto zero-shot | Ditto scratch K=100 | Ditto warm K=100 | DADER | Ditto oracle
    """
    pairs = ["P1", "P2", "P3", "P5", "P6"]
    rows = []
    for pid in pairs:
        row = {"pair_id": pid}

        # Zero-shot: warm-source checkpoint on target
        zs = df[(df.pair_id == pid) & (df.method == "ditto-zeroshot")]
        row["ditto_zeroshot_f1"] = zs.test_f1.mean() if not zs.empty else None

        # From-scratch K=100 (three seeds → mean±std)
        fs = df[(df.pair_id == pid) & (df.config == "from-scratch") & (df.k == 100)]
        row["ditto_scratch_k100_mean"] = fs.test_f1.mean() if not fs.empty else None
        row["ditto_scratch_k100_std"] = fs.test_f1.std() if len(fs) > 1 else None
        row["ditto_scratch_k100_n"] = len(fs)

        # Warm-start K=100 (three seeds → mean±std)
        ws = df[(df.pair_id == pid) & (df.method == "ditto-warmstart") & (df.k == 100)]
        row["ditto_warm_k100_mean"] = ws.test_f1.mean() if not ws.empty else None
        row["ditto_warm_k100_std"] = ws.test_f1.std() if len(ws) > 1 else None
        row["ditto_warm_k100_n"] = len(ws)

        # DADER
        dd = df[(df.pair_id == pid) & (df.paradigm == "sft") & (df.method.str.startswith("dader"))]
        row["dader_f1"] = dd.test_f1.mean() if not dd.empty else None

        rows.append(row)
    return pd.DataFrame(rows)

def build_cross_paradigm_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """main paper Sec. 4.4 Table 10: cross-paradigm comparison.

    Columns: pair | LLM K=2 random (best backbone available) | Ditto warm K=100 | DADER | paradigm best
    """
    pairs = ["P1", "P2", "P3", "P5", "P6"]
    rows = []
    for pid in pairs:
        row = {"pair_id": pid}

        # LLM K=2 random (matchgpt in-domain few-shot with k=2 random-stratified)
        llm = df[
            (df.pair_id == pid)
            & (df.method == "matchgpt_k10")
            & (df.k == 2)
        ]
        row["llm_k2_random_f1"] = llm.test_f1.mean() if not llm.empty else None
        row["llm_backbone"] = llm.backbone.iloc[0] if not llm.empty else None

        # Ditto warm K=100 (mean over seeds)
        ws = df[(df.pair_id == pid) & (df.method == "ditto-warmstart") & (df.k == 100)]
        row["ditto_warm_k100_mean"] = ws.test_f1.mean() if not ws.empty else None

        # DADER
        dd = df[(df.pair_id == pid) & (df.paradigm == "sft") & (df.method.str.startswith("dader"))]
        row["dader_f1"] = dd.test_f1.mean() if not dd.empty else None

        # Paradigm winner
        candidates = {
            "LLM": row["llm_k2_random_f1"],
            "SFT-warm": row["ditto_warm_k100_mean"],
            "SFT-DADER": row["dader_f1"],
        }
        candidates = {k: v for k, v in candidates.items() if v is not None}
        row["paradigm_best"] = max(candidates, key=candidates.get) if candidates else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------

def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def _fmt_meanstd(mean: float | None, std: float | None, n: int | None) -> str:
    if mean is None or pd.isna(mean):
        return "—"
    if std is None or pd.isna(std) or n is None or n < 2:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


PAIR_LABEL = {
    "P1": "P1 (WA→AB)",
    "P2": "P2 (Co→Wa)",
    "P3": "P3 (WA→DA)",
    "P5": "P5 (WA→AG)",
    "P6": "P6 (WA→Dirty/DA)",
    "P7": "P7 (IA→DA)",
    "P8": "P8 (IA→DS)",
    "P9": "P9 (WA→DS)",
}


def markdown_table8(t8: pd.DataFrame) -> str:
    lines = [
        "**Table 8.** Ditto and DADER F1 across five pairs. "
        "Ditto K = 100 cells report mean and standard deviation over three random seeds.",
        "",
        "| Pair | Ditto zero-shot | Ditto K=100 scratch | Ditto K=100 warm | DADER | Ditto oracle |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in t8.iterrows():
        lines.append(
            f"| {PAIR_LABEL.get(r['pair_id'], r['pair_id'])} "
            f"| {_fmt(r['ditto_zeroshot_f1'])} "
            f"| {_fmt_meanstd(r['ditto_scratch_k100_mean'], r['ditto_scratch_k100_std'], r['ditto_scratch_k100_n'])} "
            f"| {_fmt_meanstd(r['ditto_warm_k100_mean'], r['ditto_warm_k100_std'], r['ditto_warm_k100_n'])} "
            f"| {_fmt(r['dader_f1'])} "
            f"| {_fmt(r['ditto_oracle_f1'])} |"
        )
    return "\n".join(lines)


