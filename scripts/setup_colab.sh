#!/usr/bin/env bash
# setup_colab.sh — bootstrap a fresh Colab session.
#
# Idempotent: re-running is safe (no-ops on already-completed steps).
#
# Usage (from a Colab cell, after drive.mount and chdir to repo root):
#   !bash scripts/setup_colab.sh
#
# What it does, in order:
#   1. Verifies we're in the repo root (config/config.py exists)
#   2. Clones megagonlabs/ditto into ./ditto if missing
#   3. Applies three idempotent patches to Ditto source:
#        a. AdamW import (moved to torch.optim in transformers 4.30+)
#        b. apex import wrapped in try/except (apex unavailable on Colab)
#        c. --checkpoint_path warm-start support for Exp 3
#   4. Installs Python deps
#   5. Downloads spaCy + NLTK assets
#   6. Ensures our directory layout exists
#   7. Prints a "READY" banner

set -euo pipefail

REPO_ROOT="$(pwd)"
echo "[setup] repo root: ${REPO_ROOT}"

if [[ ! -f "config/config.py" ]]; then
    echo "[setup] ERROR: config/config.py not found. cd into your repo root before running this script."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Clone Ditto if missing
# ---------------------------------------------------------------------------
if [[ ! -d "ditto" ]]; then
    echo "[setup] cloning Ditto..."
    git clone --depth 1 https://github.com/megagonlabs/ditto.git
else
    echo "[setup] ditto/ already present (skip clone)"
fi

# ---------------------------------------------------------------------------
# 2. Apply idempotent patches to Ditto source
# ---------------------------------------------------------------------------
DITTO_PY="ditto/ditto_light/ditto.py"
TRAIN_PY="ditto/train_ditto.py"

# --- Patch 2a: AdamW import migrated to torch.optim ---
# Idempotency marker: presence of "from torch.optim import AdamW"
if grep -q "from torch.optim import AdamW" "${DITTO_PY}"; then
    echo "[patch 2a] AdamW import: already patched"
else
    echo "[patch 2a] applying AdamW import patch to ${DITTO_PY}"
    sed -i 's|^from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup$|from transformers import AutoModel, get_linear_schedule_with_warmup\nfrom torch.optim import AdamW|' "${DITTO_PY}"
fi

# --- Patch 2b: apex import wrapped in try/except ---
# Idempotency marker: presence of "except ImportError" on the apex line context
if grep -q "except ImportError:" "${DITTO_PY}" && grep -q "amp = None" "${DITTO_PY}"; then
    echo "[patch 2b] apex try/except: already patched"
else
    echo "[patch 2b] applying apex try/except patch to ${DITTO_PY}"
    # Replace the bare "from apex import amp" with a try/except block
    python3 - "${DITTO_PY}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = "from apex import amp\n"
new = "try:\n    from apex import amp\nexcept ImportError:\n    amp = None\n"
if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
    print(f"  [patched] {p}")
else:
    print(f"  [skip] '{old.strip()}' not found in {p}")
PY
fi

# --- Patch 2c: --checkpoint_path warm-start for Exp 3 ---
# Idempotency marker: presence of "checkpoint_path" in train_ditto.py argparse
if grep -q "\\-\\-checkpoint_path" "${TRAIN_PY}"; then
    echo "[patch 2c] --checkpoint_path argparse: already patched"
else
    echo "[patch 2c] applying --checkpoint_path argparse patch to ${TRAIN_PY}"
    # Insert the new argument right after --size (the last argparse line)
    sed -i '/parser\.add_argument("--size", type=int, default=None)/a\    parser.add_argument("--checkpoint_path", type=str, default=None,\n        help="Path to a source-trained .pt (or a directory containing one) to warm-start from. Used by Exp 3.")' "${TRAIN_PY}"
fi

if grep -q "warm-start: loading source checkpoint" "${DITTO_PY}"; then
    echo "[patch 2c-load] checkpoint-loading block: already patched"
else
    echo "[patch 2c-load] applying checkpoint-loading block to ${DITTO_PY}"
    python3 - "${DITTO_PY}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
needle = "    model = model.cuda()\n    optimizer = AdamW(model.parameters(), lr=hp.lr)\n"
if needle not in src:
    print(f"  [skip] anchor not found in {p}; manual review needed")
    sys.exit(0)
inject = (
    "    model = model.cuda()\n"
    "    # warm-start: loading source checkpoint if --checkpoint_path is set (Exp 3)\n"
    "    if getattr(hp, 'checkpoint_path', None):\n"
    "        import os as _os\n"
    "        ckpt_path = hp.checkpoint_path\n"
    "        if _os.path.isdir(ckpt_path):\n"
    "            pts = sorted(f for f in _os.listdir(ckpt_path) if f.endswith('.pt'))\n"
    "            if not pts:\n"
    "                raise FileNotFoundError(f'No .pt file in {ckpt_path}')\n"
    "            ckpt_path = _os.path.join(ckpt_path, pts[0])\n"
    "        print(f'[warm-start] loading source checkpoint: {ckpt_path}')\n"
    "        state = torch.load(ckpt_path, map_location=device)\n"
    "        if isinstance(state, dict) and 'model' in state:\n"
    "            state = state['model']\n"
    "        model.load_state_dict(state, strict=False)\n"
    "    optimizer = AdamW(model.parameters(), lr=hp.lr)\n"
)
src = src.replace(needle, inject, 1)
p.write_text(src)
print(f"  [patched] {p}")
PY
fi

# Clear any stale .pyc that would shadow our patched .py
find ditto -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Python deps
# ---------------------------------------------------------------------------
echo "[setup] installing python deps..."
pip install -q \
    "transformers>=4.30,<5" \
    "torch" \
    sentencepiece \
    jsonlines \
    nltk \
    tensorboardX \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    scipy \
    tqdm

# ---------------------------------------------------------------------------
# 4. spaCy / NLTK assets (Ditto's summarize/dk paths need these)
# ---------------------------------------------------------------------------
echo "[setup] downloading NLP assets..."
python -m spacy download en_core_web_lg -q 2>/dev/null || \
    echo "[setup]   spaCy en_core_web_lg download failed (non-fatal — only needed for --summarize/--dk)"
python - <<'PY'
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
print("[setup]   nltk stopwords + punkt ok")
PY

# ---------------------------------------------------------------------------
# 5. Ensure directory layout
# ---------------------------------------------------------------------------
python -c "from config.config import ensure_dirs; ensure_dirs(); print('[setup] dirs ok')"

# ---------------------------------------------------------------------------
# 6. Ready banner
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  READY — Colab environment bootstrapped"
echo "============================================================"
echo ""
echo "  Repo:   ${REPO_ROOT}"
echo "  Ditto:  ${REPO_ROOT}/ditto (patched)"
echo ""
echo "  Next:"
echo "    - GPU check:                python -c 'import torch; print(torch.cuda.is_available())'"
echo "    - Open notebook/ in Colab and run in numbered order (see notebook/README.md)."
echo "    - Schema-poverty ablation:  python -m src.experiments.ditto_schema_poverty_ablation --help"
echo ""