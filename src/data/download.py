"""
Download ER-Magellan datasets.

Strategy (in order):
    1. If Ditto repo is cloned and contains `data/er_magellan/<ditto_subdir>/`,
       symlink/copy from there. 
    2. Otherwise download the zip from University of Wisconsin (`zip_url`).
    3. Otherwise download each CSV individually from `uw_url` (a directory).
       Used for datasets where UW ships no zip (e.g. Textual/Company).

Each dataset produces this structure under data/raw/<type>/<dataset>/:
    train.csv, valid.csv, test.csv, tableA.csv, tableB.csv

Run directly:
    python -m src.data.download                  # all datasets in config
    python -m src.data.download Structured/Walmart-Amazon
"""

from __future__ import annotations

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from config.config import ERM_DATASETS, PATHS, ensure_dirs


# The files we need for each dataset (Ditto + analysis expect all of these).
REQUIRED_FILES = ["train.csv", "valid.csv", "test.csv", "tableA.csv", "tableB.csv"]


def _dataset_raw_dir(key: str) -> Path:
    """data/raw/Structured/Walmart-Amazon (preserves the Type/Name hierarchy)."""
    return PATHS.data_raw / key


def _have_all_files(directory: Path) -> bool:
    return directory.exists() and all((directory / f).exists() for f in REQUIRED_FILES)


def _try_copy_from_ditto_repo(key: str, meta: dict) -> bool:
    """If Ditto repo is cloned and has the dataset, copy it. Returns True on success."""
    src = PATHS.ditto_repo / meta["ditto_subdir"]
    if not src.exists():
        return False
    if not all((src / f).exists() for f in REQUIRED_FILES):
        return False

    dst = _dataset_raw_dir(key)
    dst.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED_FILES:
        shutil.copyfile(src / f, dst / f)
    print(f"  [ditto-repo] copied {key} from {src}")
    return True


def _try_download_zip(key: str, meta: dict) -> bool:
    """Download and extract the UW zip. Returns True on success."""
    url = meta.get("zip_url")
    if not url:
        return False
    dst = _dataset_raw_dir(key)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"  [uw-download] {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"  [uw-download] FAILED: {e}")
        return False

    # UW zips contain a top-level folder like `exp_data/` — we want the
    # files flattened under dst/.
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in REQUIRED_FILES:
                with zf.open(member) as src_f, open(dst / name, "wb") as out_f:
                    shutil.copyfileobj(src_f, out_f)

    if not _have_all_files(dst):
        print(f"  [uw-download] extracted but missing files under {dst}")
        return False

    print(f"  [uw-download] extracted {len(REQUIRED_FILES)} files to {dst}")
    return True


def _try_download_files(key: str, meta: dict) -> bool:
    """
    Download each REQUIRED_FILES individually from `uw_url` (a directory).
    Used when UW doesn't ship a zip for this dataset (e.g. Textual/Company).
    Returns True on success.
    """
    base = meta.get("uw_url")
    if not base:
        return False
    if not base.endswith("/"):
        base = base + "/"
    dst = _dataset_raw_dir(key)
    dst.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_FILES:
        url = base + name
        print(f"  [uw-files] {url}")
        try:
            urllib.request.urlretrieve(url, dst / name)
        except Exception as e:
            print(f"  [uw-files] FAILED on {name}: {e}")
            return False

    if not _have_all_files(dst):
        print(f"  [uw-files] missing files after fetch under {dst}")
        return False

    print(f"  [uw-files] fetched {len(REQUIRED_FILES)} files to {dst}")
    return True


def download_dataset(key: str, *, force: bool = False) -> Path:
    """
    Download one dataset. Skip if already present (unless force=True).

    Returns:
        Path to data/raw/<key>/
    """
    if key not in ERM_DATASETS:
        raise KeyError(f"Unknown dataset: {key}. Available: {list(ERM_DATASETS)}")

    meta = ERM_DATASETS[key]
    dst = _dataset_raw_dir(key)

    print(f"\n[{key}]")
    if _have_all_files(dst) and not force:
        print(f"  already present at {dst} (use force=True to redownload)")
        return dst

    if _try_copy_from_ditto_repo(key, meta):
        return dst
    if _try_download_zip(key, meta):
        return dst
    if _try_download_files(key, meta):
        return dst

    raise RuntimeError(
        f"Could not obtain {key}. Tried:\n"
        f"  - Ditto repo: {PATHS.ditto_repo / meta['ditto_subdir']}\n"
        f"  - UW zip:     {meta.get('zip_url') or '(none)'}\n"
        f"  - UW files:   {meta.get('uw_url') or '(none)'}\n"
        f"Clone Ditto first: git clone https://github.com/megagonlabs/ditto.git {PATHS.ditto_repo}"
    )


def download_all(*, force: bool = False) -> None:
    ensure_dirs()
    for key in ERM_DATASETS:
        try:
            download_dataset(key, force=force)
        except Exception as e:
            print(f"  FAILED {key}: {e}")


if __name__ == "__main__":
    ensure_dirs()
    if len(sys.argv) > 1:
        for key in sys.argv[1:]:
            download_dataset(key)
    else:
        download_all()
