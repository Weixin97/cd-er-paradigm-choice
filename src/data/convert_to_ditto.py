"""
Convert ER-Magellan CSV files into Ditto's serialized .txt format.

Input: data/raw/<key>/{train,valid,test,tableA,tableB}.csv
Output: data/processed/<key>/{train,valid,test}.txt

Ditto format (tab-separated, one pair per line):
    COL <attr> VAL <value> COL <attr> VAL <value> ... \t COL ... \t <0|1>

ER-Magellan CSVs come in two flavors — we detect which:
    (A) "Joined" format: train.csv contains ltable_<attr>/rtable_<attr> for
        every attribute, plus a `label` column. We serialize directly.
    (B) "Indexed" format: train.csv contains only id/ltable_id/rtable_id/label.
        We look up the full entity rows in tableA.csv / tableB.csv by id.

The schema (list of attribute names in order) is pulled from tableA.csv's
columns (excluding the id column), which matches how Ditto serializes.

Usage:
    python -m src.data.convert_to_ditto                  # all datasets
    python -m src.data.convert_to_ditto Structured/Walmart-Amazon
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config.config import ERM_DATASETS, PATHS, ensure_dirs


ID_COL_CANDIDATES = ("id", "_id", "ID")
LABEL_COL = "label"


def _detect_id_col(df: pd.DataFrame) -> str:
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No id column found. Columns: {list(df.columns)}")


def _schema_from_tableA(tableA: pd.DataFrame) -> list[str]:
    """Attribute names in order, excluding the id column."""
    id_col = _detect_id_col(tableA)
    return [c for c in tableA.columns if c != id_col]


def _serialize_entity(row: pd.Series, attrs: list[str]) -> str:
    """Build 'COL <attr> VAL <value> COL <attr> VAL <value> ...'."""
    parts: list[str] = []
    for attr in attrs:
        val = row.get(attr, "")
        if pd.isna(val):
            val = ""
        # Ditto tokenizes on whitespace — collapse tabs/newlines in values
        val_str = str(val).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        parts.append(f"COL {attr} VAL {val_str}")
    return " ".join(parts)


def _is_joined_format(df: pd.DataFrame) -> bool:
    """Joined = has ltable_<attr>/rtable_<attr> columns beyond just the ids."""
    has_lprefix = any(c.startswith("ltable_") and c != "ltable_id" for c in df.columns)
    has_rprefix = any(c.startswith("rtable_") and c != "rtable_id" for c in df.columns)
    return has_lprefix and has_rprefix


def _convert_joined(
    df: pd.DataFrame,
    attrs: list[str],
) -> list[str]:
    """Rows already have ltable_<attr>/rtable_<attr>. Build Ditto lines."""
    lines = []
    for _, row in df.iterrows():
        left_row = pd.Series({a: row.get(f"ltable_{a}", "") for a in attrs})
        right_row = pd.Series({a: row.get(f"rtable_{a}", "") for a in attrs})
        left = _serialize_entity(left_row, attrs)
        right = _serialize_entity(right_row, attrs)
        label = int(row[LABEL_COL])
        lines.append(f"{left}\t{right}\t{label}")
    return lines


def _convert_indexed(
    df: pd.DataFrame,
    tableA: pd.DataFrame,
    tableB: pd.DataFrame,
    attrs: list[str],
) -> list[str]:
    """Rows are (ltable_id, rtable_id, label); look up full entities."""
    id_col_a = _detect_id_col(tableA)
    id_col_b = _detect_id_col(tableB)
    idx_a = tableA.set_index(id_col_a)
    idx_b = tableB.set_index(id_col_b)

    lines = []
    for _, row in df.iterrows():
        try:
            left_row = idx_a.loc[row["ltable_id"]]
            right_row = idx_b.loc[row["rtable_id"]]
        except KeyError as e:
            print(f"  skipping pair with missing id: {e}")
            continue
        left = _serialize_entity(left_row, attrs)
        right = _serialize_entity(right_row, attrs)
        label = int(row[LABEL_COL])
        lines.append(f"{left}\t{right}\t{label}")
    return lines


def convert_dataset(key: str, *, force: bool = False) -> Path:
    """
    Convert one dataset's {train,valid,test}.csv -> .txt under data/processed.

    Returns:
        Path to the output directory.
    """
    src = PATHS.data_raw / key
    dst = PATHS.data_processed / key
    dst.mkdir(parents=True, exist_ok=True)

    print(f"\n[{key}]")
    if not src.exists():
        raise FileNotFoundError(f"Raw data not found: {src}. Run download first.")

    tableA = pd.read_csv(src / "tableA.csv")
    tableB = pd.read_csv(src / "tableB.csv")
    attrs = _schema_from_tableA(tableA)
    print(f"  schema: {attrs}")

    for split in ("train", "valid", "test"):
        out = dst / f"{split}.txt"
        if out.exists() and not force:
            print(f"  {split}.txt already exists (skip)")
            continue

        df = pd.read_csv(src / f"{split}.csv")
        if _is_joined_format(df):
            lines = _convert_joined(df, attrs)
            fmt = "joined"
        else:
            lines = _convert_indexed(df, tableA, tableB, attrs)
            fmt = "indexed"

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  [{fmt}] {split}: {len(lines)} pairs -> {out}")

    return dst


def convert_all(*, force: bool = False) -> None:
    ensure_dirs()
    for key in ERM_DATASETS:
        try:
            convert_dataset(key, force=force)
        except Exception as e:
            print(f"  FAILED {key}: {e}")


if __name__ == "__main__":
    ensure_dirs()
    if len(sys.argv) > 1:
        for key in sys.argv[1:]:
            convert_dataset(key)
    else:
        convert_all()
