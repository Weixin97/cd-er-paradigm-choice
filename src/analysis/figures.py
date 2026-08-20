"""
main paper — figure generators.

Reads metrics.json files under `results/runs/sparsity-ablation/` (LLM cells) and
`results/runs/sparsity-ablation-sft/` (Ditto cells) and produces two figures for
the paper draft.

Figure 2 — F1 vs richness curves per target
    One panel per target (Amazon-Google, Abt-Buy, DBLP-ACM), level on x-axis,
    F1 on y-axis, two lines per panel: LLM K = 2 random-stratified and Ditto
    warm-start K = 100. Error bars are the three-seed standard deviation.

Figure 3 — Cost/latency tradeoff
    Two-panel scatter: F1 vs per-record cost (left) and F1 vs per-record
    latency (right). One marker per (paradigm, target) combination.

Usage:
    python -m src.analysis.figures --figure 2
    python -m src.analysis.figures --figure 3
    python -m src.analysis.figures --all
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.config import PATHS


LLM_ROOT = PATHS.results_runs / "sparsity-ablation"
SFT_ROOT = PATHS.results_runs / "sparsity-ablation-sft"
FIG_ROOT = PATHS.results_figures / "main_matrix"

TARGET_LABELS = {
    "Structured/Amazon-Google": "Amazon-Google (structured product)",
    "Textual/Abt-Buy":          "Abt-Buy (textual product)",
    "Structured/DBLP-ACM":      "DBLP-ACM (citation)",
}
TARGET_ORDER = [
    "Structured/Amazon-Google",
    "Textual/Abt-Buy",
    "Structured/DBLP-ACM",
]

# DeepInfra 2026 pricing for Llama-3.3-70B
COST_IN_PER_M = 0.23
COST_OUT_PER_M = 0.40

# ---------------------------------------------------------------------------
# Figure 4 configuration / Table 10b as grouped bar chart
# ---------------------------------------------------------------------------
LORA_ROOT = PATHS.results_runs / "main_matrix" / "llm-lora"
PROMPT_ROOT = PATHS.results_runs / "main_matrix"
DITTO_WS_ROOT = PATHS.results_runs / "ditto-warmstart"

# (label, source, target, lora_key)
LORA_PAIRS = [
    ("P1 (WA→AB)", "Structured/Walmart-Amazon", "Textual/Abt-Buy",          "P1_WA_AB"),
    ("P2 (Co→Wa)", "wdc/computers",              "wdc/watches",              "P2_Co_Wa"),
    ("P3 (WA→DA)", "Structured/Walmart-Amazon", "Structured/DBLP-ACM",      "P3_WA_DA"),
    ("P4 (WA→AG)", "Structured/Walmart-Amazon", "Structured/Amazon-Google", "P4_WA_AG"),
]


def _load_llm_by_target_rung() -> dict[tuple[str, int], list[dict]]:
    """Aggregate LLM cells by (target, rung), unperturbed only."""
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for f in sorted(LLM_ROOT.rglob("metrics.json")):
        d = json.loads(f.read_text())
        if d.get("method") != "random_k2":
            continue
        if d.get("perturbation") != "none":
            continue
        out[(d["target_dataset"], d["rung"])].append(d)
    return out


def _load_sft_by_target_rung() -> dict[tuple[str, int], list[dict]]:
    """Aggregate Ditto cells by (target, rung), unperturbed only."""
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for f in sorted(SFT_ROOT.rglob("metrics.json")):
        d = json.loads(f.read_text())
        if d.get("perturbation") != "none":
            continue
        out[(d["target_dataset"], d["rung"])].append(d)
    return out


def _mean_sd(cells: list[dict], field: str) -> tuple[float | None, float]:
    vals = [d.get(field) for d in cells if d.get(field) is not None]
    if not vals:
        return None, 0.0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def figure_2(out_dir: Path) -> None:
    """F1 vs richness curves — one panel per target, two paradigms per panel."""
    llm = _load_llm_by_target_rung()
    sft = _load_sft_by_target_rung()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, target in zip(axes, TARGET_ORDER):
        rungs = sorted({r for (t, r) in llm.keys() if t == target}
                       | {r for (t, r) in sft.keys() if t == target})

        # LLM line
        llm_means, llm_sds = [], []
        for r in rungs:
            m, s = _mean_sd(llm.get((target, r), []), "test_f1")
            llm_means.append(m); llm_sds.append(s)
        # SFT line
        sft_means, sft_sds = [], []
        for r in rungs:
            m, s = _mean_sd(sft.get((target, r), []), "test_f1")
            sft_means.append(m); sft_sds.append(s)

        rungs_arr = np.array(rungs, dtype=float)
        llm_means_arr = np.array([v if v is not None else np.nan for v in llm_means])
        sft_means_arr = np.array([v if v is not None else np.nan for v in sft_means])

        ax.errorbar(rungs_arr, llm_means_arr, yerr=llm_sds, marker="o",
                    color="#1f77b4", linewidth=2, capsize=3, label="LLM K=2 (Llama-3.3-70B)")
        ax.errorbar(rungs_arr, sft_means_arr, yerr=sft_sds, marker="s",
                    color="#ff7f0e", linewidth=2, capsize=3, label="Ditto warm-start K=100")

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xlabel("Level (0 = full schema → 4 = truncated stub)")
        ax.set_xticks(rungs)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

    axes[0].set_ylabel("F1 (three-seed mean ± sd)")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig2_f1_vs_richness.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[saved] {out_path}")
    plt.close(fig)


def figure_3(out_dir: Path) -> None:
    """Cost/latency tradeoff scatter."""
    llm = _load_llm_by_target_rung()

    # Per-target LLM stats at rung 0 (full-schema headline)
    llm_pts = []
    for target in TARGET_ORDER:
        cells = llm.get((target, 0), [])
        if not cells:
            continue
        f1_mean, _ = _mean_sd(cells, "test_f1")
        # cost per 1K records
        pt_mean = st.mean([d.get("prompt_tokens_mean", 0) for d in cells])
        ct_mean = st.mean([d.get("completion_tokens_mean", 0) for d in cells])
        cost_per_1k = (pt_mean * COST_IN_PER_M + ct_mean * COST_OUT_PER_M) / 1000
        # latency per record (s → ms)
        n_test = cells[0].get("n_test_pairs", 1) or 1
        lat_ms = 1000 * st.mean([d.get("api_latency_sec", 0) / (d.get("n_test_pairs", 1) or 1)
                                 for d in cells])
        llm_pts.append((TARGET_LABELS[target], cost_per_1k, lat_ms, f1_mean))

    # Ditto: full-schema F1 per target (from Ditto ablation rung 0), cost/latency fixed
    sft = _load_sft_by_target_rung()
    sft_pts = []
    for target in TARGET_ORDER:
        cells = sft.get((target, 0), [])
        if not cells:
            continue
        f1_mean, _ = _mean_sd(cells, "test_f1")
        # Ditto amortised cost/latency (approximate, from §4.3)
        cost_per_1k = 0.001
        lat_ms = 8.0
        sft_pts.append((TARGET_LABELS[target], cost_per_1k, lat_ms, f1_mean))

    fig, (ax_cost, ax_lat) = plt.subplots(1, 2, figsize=(13, 6.5))

    colors = {"LLM K=2": "#1f77b4", "Ditto WS K=100": "#ff7f0e"}
    markers = {
        TARGET_LABELS["Structured/Amazon-Google"]: "o",
        TARGET_LABELS["Textual/Abt-Buy"]:          "^",
        TARGET_LABELS["Structured/DBLP-ACM"]:      "s",
    }

    # LEFT panel — cost vs F1
    # LLM points sit on the right edge → labels go LEFT so they stay in-plot
    for label, cost, _, f1 in llm_pts:
        ax_cost.scatter(cost, f1, s=150, marker=markers[label], color=colors["LLM K=2"],
                        edgecolors="black", linewidth=1)
        ax_cost.annotate(f"LLM · {label.split(' (')[0]}", (cost, f1),
                         xytext=(-14, 0), textcoords="offset points",
                         fontsize=8, va="center", ha="right")
    # SFT points sit on the left edge → labels go RIGHT
    for label, cost, _, f1 in sft_pts:
        ax_cost.scatter(cost, f1, s=150, marker=markers[label], color=colors["Ditto WS K=100"],
                        edgecolors="black", linewidth=1)
        ax_cost.annotate(f"Ditto · {label.split(' (')[0]}", (cost, f1),
                         xytext=(14, 0), textcoords="offset points",
                         fontsize=8, va="center", ha="left")

    ax_cost.set_xscale("log")
    ax_cost.set_xlabel("Per-record inference cost, USD / 1K records (log scale)")
    ax_cost.set_ylabel("F1 (full-schema, three-seed mean)")
    ax_cost.set_ylim(0.55, 1.0)
    ax_cost.grid(True, alpha=0.3, which="both")
    ax_cost.set_title("(a) Cost tradeoff", loc="left", fontsize=11)

    # RIGHT panel — latency vs F1 (same edge convention as left panel)
    for label, _, lat, f1 in llm_pts:
        ax_lat.scatter(lat, f1, s=150, marker=markers[label], color=colors["LLM K=2"],
                       edgecolors="black", linewidth=1)
        ax_lat.annotate(f"LLM · {label.split(' (')[0]}", (lat, f1),
                        xytext=(-14, 0), textcoords="offset points",
                        fontsize=8, va="center", ha="right")
    for label, _, lat, f1 in sft_pts:
        ax_lat.scatter(lat, f1, s=150, marker=markers[label], color=colors["Ditto WS K=100"],
                       edgecolors="black", linewidth=1)
        ax_lat.annotate(f"Ditto · {label.split(' (')[0]}", (lat, f1),
                        xytext=(14, 0), textcoords="offset points",
                        fontsize=8, va="center", ha="left")

    ax_lat.set_xscale("log")
    ax_lat.set_xlabel("Per-record inference latency, ms (log scale)")
    ax_lat.set_ylim(0.55, 1.0)
    ax_lat.grid(True, alpha=0.3, which="both")
    ax_lat.set_title("(b) Latency tradeoff", loc="left", fontsize=11)

    # Reserve top strip for the shared legend, then apply layout
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Shared paradigm legend above both panels
    handles = [
        plt.Line2D([0], [0], marker="o", markerfacecolor=colors["LLM K=2"],
                    color="w", markeredgecolor="black", markersize=10, label="LLM K=2 (Llama-3.3-70B)"),
        plt.Line2D([0], [0], marker="o", markerfacecolor=colors["Ditto WS K=100"],
                    color="w", markeredgecolor="black", markersize=10, label="Ditto warm-start K=100"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, fontsize=10, frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig3_cost_latency_tradeoff.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[saved] {out_path}")
    plt.close(fig)


def _read_seeds(root: Path, glob: str) -> tuple[float | None, float]:
    """Read `test_f1` from all metrics.json under `root` matching `glob`, return (mean, sd)."""
    vals = []
    for f in sorted(root.glob(glob)):
        try:
            vals.append(float(json.loads(f.read_text())["test_f1"]))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    if not vals:
        return None, 0.0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def figure_4(out_dir: Path) -> None:
    """Table 10b as a grouped bar chart: LoRA K=100 vs three references across P1–P4."""
    n_pairs = len(LORA_PAIRS)
    n_methods = 4
    method_labels = [
        "LLM K=2 on Llama-3.1-8B",
        "LLM LoRA K=100 on Llama-3.1-8B",
        "Ditto warm-start K=100 (RoBERTa)",
        "LLM K=2 on Llama-3.3-70B",
    ]
    method_colors = ["#a6cee3", "#1f78b4", "#ff7f0e", "#33a02c"]

    src_to_dir = lambda s: s.replace("/", "__")
    tgt_to_dir = lambda t: t.replace("/", "__")

    all_means = [[None] * n_pairs for _ in range(n_methods)]
    all_sds   = [[0.0]  * n_pairs for _ in range(n_methods)]

    for i, (label, src, tgt, lora_key) in enumerate(LORA_PAIRS):
        # LLM K=2 random-stratified on Llama-3.1-8B
        all_means[0][i], all_sds[0][i] = _read_seeds(
            PROMPT_ROOT / "llama-3.1-8b" / tgt_to_dir(tgt) / "random_k2",
            "seed_*/metrics.json",
        )
        # LLM LoRA K=100 on Llama-3.1-8B
        all_means[1][i], all_sds[1][i] = _read_seeds(
            LORA_ROOT / lora_key, "k_100_seed_*/metrics.json",
        )
        # Ditto warm-start K=100 (RoBERTa)
        all_means[2][i], all_sds[2][i] = _read_seeds(
            DITTO_WS_ROOT / f"{src_to_dir(src)}__from__{tgt_to_dir(tgt)}",
            "k_100_seed_*/metrics.json",
        )
        # LLM K=2 random-stratified on Llama-3.3-70B
        all_means[3][i], all_sds[3][i] = _read_seeds(
            PROMPT_ROOT / "llama-3.3-70b" / tgt_to_dir(tgt) / "random_k2",
            "seed_*/metrics.json",
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_w = 0.19
    x_base = np.arange(n_pairs)

    for m_idx in range(n_methods):
        means = [v if v is not None else 0 for v in all_means[m_idx]]
        ax.bar(x_base + (m_idx - 1.5) * bar_w, means, bar_w,
               yerr=all_sds[m_idx], capsize=3,
               label=method_labels[m_idx], color=method_colors[m_idx],
               edgecolor="black", linewidth=0.4)

    # Mark P4 LoRA seed instability
    p4_idx = 3
    if all_means[1][p4_idx] is not None:
        ax.annotate(
            "seed-unstable",
            xy=(x_base[p4_idx] - 0.5 * bar_w, all_means[1][p4_idx] + all_sds[1][p4_idx] + 0.04),
            fontsize=8, ha="center", color="#1f78b4",
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([label for (label, *_) in LORA_PAIRS])
    ax.set_ylabel("F1 (three-seed mean ± sd)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=2, fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig4_lora_vs_references.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[saved] {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, choices=[2, 3, 4],
                        help="Which figure to generate (default: --all)")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--output", type=str, default=str(FIG_ROOT),
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.all or args.figure == 2:
        figure_2(out_dir)
    if args.all or args.figure == 3:
        figure_3(out_dir)
    if args.all or args.figure == 4:
        figure_4(out_dir)

    if not args.all and args.figure is None:
        parser.print_help()


if __name__ == "__main__":
    main()
