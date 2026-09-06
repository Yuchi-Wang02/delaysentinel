"""Figures for the case study: ``python -m delaysentinel.eda --out docs/figures``.

Each figure is post-hoc exploratory analysis done in September 2026; it should have
preceded training in September 2025.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .data import load_csv

ROOT = Path(__file__).resolve().parents[2]

# Chart chrome (light surface) and two validated categorical hues (blue, orange).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
NOTE = "post-hoc EDA (Sept 2026); should have preceded training (Sept 2025)"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def _footer(fig, text: str = NOTE) -> None:
    fig.text(0.01, 0.01, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def fig_latlon(csv: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for label, color, name in ((0, BLUE, "not delayed"), (1, ORANGE, "delayed")):
        sub = csv[csv["Logistics_Delay"] == label]
        ax.scatter(
            sub["Longitude"],
            sub["Latitude"],
            s=9,
            color=color,
            alpha=0.75,
            linewidths=0,
            label=f"{name} (n={len(sub)})",
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("1,000 'truck' snapshots are uniform over the whole globe", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, markerscale=2)
    ax.text(-176, -84, "no land mask, no lanes, no depots: the coordinates are synthetic", fontsize=8, color=INK2)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_latlon.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_crosstab(csv: pd.DataFrame, out: Path) -> Path:
    statuses = ["Delayed", "Delivered", "In Transit"]
    traffics = ["Heavy", "Detour", "Clear"]
    rate = np.zeros((len(statuses), len(traffics)))
    count = np.zeros_like(rate, dtype=int)
    for i, s in enumerate(statuses):
        for j, t in enumerate(traffics):
            sub = csv[(csv["Shipment_Status"] == s) & (csv["Traffic_Status"] == t)]
            count[i, j] = len(sub)
            rate[i, j] = sub["Logistics_Delay"].mean() if len(sub) else np.nan
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(rate, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(traffics)), traffics)
    ax.set_yticks(range(len(statuses)), statuses)
    ax.set_xlabel("Traffic_Status")
    ax.set_ylabel("Shipment_Status")
    ax.grid(False)
    for i in range(len(statuses)):
        for j in range(len(traffics)):
            txt = f"{rate[i, j]:.0%}\nn={count[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color="white" if rate[i, j] > 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("share of rows labelled delayed", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title("The label is 100% or 0% in every cell: Delayed OR Heavy", loc="left", fontsize=11)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_crosstab.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_probes(eval_json: dict, out: Path) -> Path:
    rows = []
    cf = eval_json.get("counterfactual_probe", {})
    labels = {
        "status_Delayed_to_InTransit": "Delayed → In Transit (expect 0)",
        "traffic_Heavy_to_Clear": "Heavy → Clear (expect 0)",
        "negatives_traffic_to_Heavy": "negatives: traffic → Heavy (expect 1)",
        "negatives_status_to_Delayed": "negatives: status → Delayed (expect 1)",
        "negatives_waiting60_temp30": "negatives: wait=60, temp=30 (expect unchanged)",
        "positives_waiting10_temp18": "positives: wait=10, temp=18 (expect unchanged)",
    }
    for key, name in labels.items():
        r = cf.get(key)
        if not r or r.get("matches_expected") is None:
            continue
        rows.append((name, r["matches_expected"], r["n"]))
    if not rows:
        raise ValueError("eval.json has no counterfactual probes")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    shares = [m / n for _, m, n in rows]
    ax.barh(y, shares, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=1)
    for yi, (_name, m, n), s in zip(y, rows, shares, strict=False):
        ax.text(
            min(s, 1) - 0.01, yi, f"{m}/{n}", va="center", ha="right", fontsize=9, color="white" if s > 0.15 else INK
        )
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of edited prompts whose prediction matches the rule's expectation")
    ax.set_title("Counterfactual probes: the two rule fields decide, the others do not", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    _footer(fig, "greedy decoding on edited copies of the 200 frozen test prompts; numbers from results/eval.json")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "fig_probes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_training(runs_dir: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = {"sc904": BLUE, "sc920": ORANGE, "sc9201": "#1baf7a"}
    names = {"sc904": "30 epochs (published)", "sc920": "100 epochs", "sc9201": "65 epochs"}
    for run in ("sc904", "sc9201", "sc920"):
        with open(runs_dir / run / "trainer_state.json", encoding="utf-8") as handle:
            state = json.load(handle)
        tr = [(e["step"], e["loss"]) for e in state["log_history"] if "loss" in e]
        ev = [(e["step"], e["eval_loss"] * 1e6) for e in state["log_history"] if "eval_loss" in e]
        axes[0].plot([s for s, _ in tr], [lab for _, lab in tr], lw=1.4, color=colors[run], label=names[run])
        axes[1].plot(
            [s for s, _ in ev], [lab for _, lab in ev], lw=1.4, marker="o", ms=3, color=colors[run], label=names[run]
        )
    axes[0].set_xlim(0, 300)
    axes[0].set_xlabel("optimizer step (first 300 shown)")
    axes[0].set_ylabel("training loss (assistant tokens)")
    axes[0].set_title("Training loss is exactly 0.0 within 50–70 steps", loc="left", fontsize=10.5)  # noqa: RUF001
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("eval loss × 1e-6 (first 20 rows of the eval file)")  # noqa: RUF001
    axes[1].set_title("Eval loss is flat after ~epoch 20", loc="left", fontsize=10.5)
    axes[1].legend(frameon=False, fontsize=8.5)
    _footer(
        fig, "from runs/*/trainer_state.json; the upstream trainer evaluates only the first 20 rows of the eval file"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out / "fig_training.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_positive_control(pc: dict, out: Path) -> Path:
    """Precision-recall at fixed thresholds and a reliability diagram for the Olist positive control."""
    models = pc["models"]
    prevalence = pc["data"]["test_late_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    for name, color, label in (
        ("logistic_regression", BLUE, "logistic regression"),
        ("hist_gradient_boosting", ORANGE, "hist. gradient boosting"),
    ):
        curve = [r for r in models[name]["pr_curve"] if r["precision"] is not None and r["flagged_share"] > 0]
        ax.plot([r["recall"] for r in curve], [r["precision"] for r in curve], lw=1.6, color=color, label=label)
    ax.axhline(prevalence, color=MUTED, lw=1, ls="--")
    ax.text(0.99, prevalence + 0.006, f"prevalence {prevalence:.3f}", ha="right", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.4, ax.get_ylim()[1]))
    ax.set_xlabel("recall of late orders")
    ax.set_ylabel("precision among flagged orders")
    ax.set_title("Olist test period: precision-recall at thresholds 0.02-0.60", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[1]
    rel = [r for r in models["logistic_regression"]["reliability"] if r["n"] >= 30]
    ax.plot([0, 0.5], [0, 0.5], color=MUTED, lw=1, ls="--")
    ax.plot([r["mean_pred"] for r in rel], [r["observed_rate"] for r in rel], marker="o", ms=4, lw=1.6, color=BLUE)
    for r in rel:
        ax.annotate(
            f"n={r['n']:,}",
            (r["mean_pred"], r["observed_rate"]),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=7.5,
            color=INK2,
        )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("mean predicted probability (bins with n >= 30)")
    ax.set_ylabel("observed late rate")
    ax.set_title("Logistic regression reliability (test period)", loc="left", fontsize=10.5)
    _footer(
        fig, "Olist orders, train < 2018-03-01, test >= 2018-03-01; numbers from results/olist_positive_control.json"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out / "fig_positive_control.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(ROOT / "data" / "smart_logistics_dataset.csv"))
    parser.add_argument("--eval", default=str(ROOT / "results" / "eval.json"))
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = load_csv(args.csv)
    paths = [fig_latlon(csv, out), fig_crosstab(csv, out), fig_training(Path(args.runs), out)]
    if Path(args.eval).exists():
        with open(args.eval, encoding="utf-8") as handle:
            paths.append(fig_probes(json.load(handle), out))
    pc_path = Path(args.eval).with_name("olist_positive_control.json")
    if pc_path.exists():
        with open(pc_path, encoding="utf-8") as handle:
            paths.append(fig_positive_control(json.load(handle), out))
    for p in paths:
        print("wrote", p)
    return paths


if __name__ == "__main__":
    main()
