"""Draw the README figures from reports/*.csv.

Reads the saved sweep output only, so this is free to run and the pictures cannot
drift from the tables.  The figures exist to separate quantities the field tends
to report as one: detection quality, ring recovery, and whether the explanation
points at the ring's own edges.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

DETECTION_COLUMNS = [
    "topology", "camouflage", "model",
    "auc", "auc_std", "ap", "ap_std", "ring_recall", "ring_recall_std",
]
FAITH_COLUMNS = [
    "topology", "camouflage", "model", "explainer", "detected", "k_mode",
    "precision", "precision_std", "random_expectation", "random_expectation_std",
    "lift", "lift_std", "n_candidates", "n_candidates_std",
    "n_relevant", "n_relevant_std",
]

# One colour per explainer, held across every figure, so `ig` is the same blue
# wherever a reader meets it. The random explainer is the control and stays grey.
EXPLAINERS = ["ig", "grad", "gnnexplainer", "random"]
EXPLAINER_COLOURS = {
    "ig": PALETTE[0],
    "grad": PALETTE[3],
    "gnnexplainer": PALETTE[4],
    "random": PALETTE[5],
}
TOPOLOGY_COLOURS = {
    "clique": PALETTE[0],
    "bipartite": PALETTE[3],
    "cycle": PALETTE[2],
    "star": PALETTE[1],
}
MODEL_COLOURS = {"gcn": PALETTE[0], "sage": PALETTE[3], "mlp": PALETTE[5]}

CAMO_AXIS = "camouflage (cover edges per unit of in-ring degree)"


def _load(name: str, columns: list[str]) -> pd.DataFrame:
    """Read one of the two-row-header summaries and give it flat column names."""
    table = pd.read_csv(REPORTS / name, header=[0, 1], skiprows=[2])
    table.columns = columns
    return table


def _detected_gcn() -> pd.DataFrame:
    """Faithfulness rows for the GCN on fraud nodes it actually detected."""
    table = _load("faithfulness_summary.csv", FAITH_COLUMNS).dropna(subset=["lift"])
    return table[(table.model == "gcn") & (table.detected.astype(str) == "1")]


def _end_label(ax, x, y, text: str, colour: str, dy: float = 0.0) -> None:
    """Label a line at its right end, so the panel needs no legend box."""
    ax.annotate(text, xy=(x, y), xytext=(7, dy), textcoords="offset points",
                color=colour, fontsize=9.3, va="center", ha="left")


def detection_vs_recovery(out: Path) -> Path:
    """Show AUC holding up while ring recovery collapses.

    Camouflage is the x-axis because it is the only knob that separates the two:
    a model can keep ranking fraudulent nodes above clean ones while finding none
    of the rings that make the fraud worth investigating.
    """
    table = _load("detection_summary.csv", DETECTION_COLUMNS)
    grouped = (
        table[table.model == "gcn"]
        .groupby("camouflage")[["auc", "ap", "ring_recall"]].mean()
    )
    runs = pd.read_csv(REPORTS / "detection_raw.csv").query("model == 'gcn'")
    x = grouped.index.to_numpy()

    figure, ax = plt.subplots(figsize=(8.6, 5.0))
    for column, label, colour in [
        ("auc", "node AUC", PALETTE[0]),
        ("ap", "average precision", PALETTE[3]),
        ("ring_recall", "ring recovery", PALETTE[1]),
    ]:
        ax.plot(runs.camouflage, runs[column], "o", markersize=2.8, alpha=0.22,
                color=colour, zorder=1)
        ax.plot(x, grouped[column], "o-", color=colour, label=label, lw=2.2, zorder=3)
    ax.axhline(0.5, color="#bbbbbb", lw=0.9, ls=":", zorder=0)
    ax.annotate(
        f"AUC still {grouped.auc.iloc[-1]:.2f},\n"
        f"{grouped.ring_recall.iloc[-1] * 100:.0f}% of rings found",
        xy=(x[-1], grouped.ring_recall.iloc[-1]),
        xytext=(1.12, 0.58),
        fontsize=9.3,
        color="#444444",
        arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 0.9},
    )
    ax.set_xlabel(CAMO_AXIS)
    ax.set_ylabel("score (fraction, 0 to 1)")
    ax.set_ylim(-0.03, 1.05)
    titled(ax, "The score barely moves while the rings disappear",
           "GCN, mean over 4 ring topologies x 5 seeds; faint dots are the 20 runs behind each mean")
    ax.legend(loc="lower left")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def faithfulness(out: Path) -> Path:
    """Explainer precision against the random-edge expectation it must be read against.

    The random explainer is the control: it should land on a lift of exactly 1, and
    any deviation would mean the null is mis-specified rather than that random
    edge selection works.
    """
    detected = _detected_gcn()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    summary = (
        detected.groupby("explainer")[["precision", "random_expectation"]]
        .mean().loc[EXPLAINERS]
    )
    base = np.arange(len(EXPLAINERS))
    left.bar(base - 0.2, summary.precision, 0.4, label="explainer precision",
             color=[EXPLAINER_COLOURS[e] for e in EXPLAINERS], edgecolor="none")
    left.bar(base + 0.2, summary.random_expectation, 0.4,
             label="random-edge expectation", color="none", edgecolor="#555555",
             lw=1.1, hatch="///")
    left.set_xticks(base)
    left.set_xticklabels(EXPLAINERS)
    left.set_ylim(0, 0.52)
    left.set_ylabel("ring edges among the top-k (fraction)")
    titled(left, "Even the best explainer is close to the null",
           "IG names 41% ring edges where chance gives 23%; GCN, all 6 budgets")
    left.legend(loc="upper right")

    by_camouflage = (
        detected.groupby(["camouflage", "explainer"])["lift"].mean().unstack()
    )
    for explainer in EXPLAINERS:
        style = "s--" if explainer == "random" else "o-"
        label = "random (control)" if explainer == "random" else explainer
        right.plot(by_camouflage.index, by_camouflage[explainer], style,
                   color=EXPLAINER_COLOURS[explainer], lw=1.9)
        _end_label(right, by_camouflage.index[-1], by_camouflage[explainer].iloc[-1],
                   label, EXPLAINER_COLOURS[explainer])
    right.axhline(1.0, color="#666666", lw=1.0, zorder=0)
    right.set_xlim(-0.1, 2.75)
    right.set_xlabel(CAMO_AXIS)
    right.set_ylabel("lift over the random-edge null (x, 1.0 = chance)")
    titled(right, "Lift rises as the null thins faster than the explainer",
           "same runs; the control sits on 1.0, which is the check on the null")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def by_topology(out: Path) -> Path:
    """Ring recovery per ring shape, which is what camouflage acts on.

    A clique of colluders is a dense subgraph and survives longest, still 10%
    recovered at camouflage 2.0 where every other shape is at zero. Star and cycle
    are the sparsest and are gone by camouflage 1.0. Averaging over topologies, as
    the headline figure does, hides a spread that wide.
    """
    table = _load("detection_summary.csv", DETECTION_COLUMNS)
    gcn = table[table.model == "gcn"]
    topologies = sorted(gcn.topology.unique())

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    for topology in topologies:
        rows = gcn[gcn.topology == topology].sort_values("camouflage")
        colour = TOPOLOGY_COLOURS[topology]
        left.plot(rows.camouflage, rows.auc, "o-", lw=1.9, color=colour, label=topology)
        right.plot(rows.camouflage, rows.ring_recall, "o-", lw=1.9, color=colour,
                   label=topology)
    for ax, ylabel in ((left, "node AUC (0.5 = chance)"),
                       (right, "rings recovered (fraction of planted rings)")):
        ax.set_xlabel(CAMO_AXIS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.05)
    titled(left, "Every ring shape keeps a node AUC that reads as working",
           "GCN, mean of 5 seeds per point")
    titled(right, "Only the dense clique is still recoverable at camouflage 2.0",
           "the same runs, scored on whether the planted ring was surfaced")
    left.legend(loc="lower left", title="ring topology", title_fontsize=9)

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def model_comparison(out: Path) -> Path:
    """The three models, including the one that cannot see the graph.

    MLP reads node features only. Whatever it scores is available without any
    structure at all, so it is the floor the graph models have to beat before any
    of this is about collusion.
    """
    table = _load("detection_summary.csv", DETECTION_COLUMNS)
    models = ["gcn", "sage", "mlp"]
    panels = [
        ("auc", "node AUC (0.5 = chance)",
         "Graph models beat the feature-only floor",
         "mean over 4 ring topologies x 5 seeds"),
        ("ap", "average precision (fraction, 0 to 1)",
         "Average precision falls faster than AUC",
         "the ranking degrades faster than AUC admits"),
        ("ring_recall", "rings recovered (fraction of planted rings)",
         "The MLP never recovers a single ring",
         "the same runs, scored on the planted rings"),
    ]

    labels = {"gcn": "GCN", "sage": "GraphSAGE", "mlp": "MLP (features only)"}

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True)
    for ax, (column, ylabel, title, subtitle) in zip(axes, panels, strict=True):
        for model in models:
            rows = table[table.model == model].groupby("camouflage")[column].mean()
            style = "s--" if model == "mlp" else "o-"
            ax.plot(rows.index, rows.values, style, color=MODEL_COLOURS[model], lw=2,
                    label=labels[model])
        ax.set_xlabel(CAMO_AXIS, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.05)
        titled(ax, title, subtitle)
    axes[0].legend(loc="lower left")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def budget_sensitivity(out: Path) -> Path:
    """Faithfulness against how many edges the explainer is allowed to name.

    ``oracle`` hands the explainer the true number of ring edges, which is the
    budget the original version of this repo measured at and is not available at
    inference time. The fixed budgets are what an investigator actually gets.
    """
    detected = _detected_gcn()
    order = ["k1", "k3", "k5", "k10", "k20", "oracle"]
    positions = np.arange(len(order))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for explainer in EXPLAINERS:
        rows = (
            detected[detected.explainer == explainer]
            .groupby("k_mode")[["precision", "lift"]].mean().reindex(order)
        )
        style = "s--" if explainer == "random" else "o-"
        label = "random (control)" if explainer == "random" else explainer
        colour = EXPLAINER_COLOURS[explainer]
        for ax, column in ((left, "precision"), (right, "lift")):
            ax.plot(positions, rows[column], style, color=colour, lw=1.9)
            _end_label(ax, positions[-1], rows[column].iloc[-1], label, colour)
    for ax, ylabel, title, subtitle in (
        (left, "ring edges among the top-k (fraction)",
         "Raw precision looks better at a tighter budget",
         "GCN, fraud nodes it detected, mean over 16 topology x camouflage cells"),
        (right, "lift over the random-edge null (x, 1.0 = chance)",
         "Lift stays flat, because the null tightens with the budget",
         "oracle is the true ring size, which nobody has at inference time"),
    ):
        ax.set_xticks(positions)
        ax.set_xticklabels(order)
        ax.set_xlim(-0.25, 6.6)
        ax.set_xlabel("explanation budget (edges the explainer may name)")
        ax.set_ylabel(ylabel)
        titled(ax, title, subtitle)
    right.axhline(1.0, color="#666666", lw=1.0, zorder=0)

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def _shrink_gif(path: Path, colours: int = 200) -> Path:
    """Rewrite every frame onto one shared palette, which roughly halves the file.

    The frames are mostly white, so a small palette spends its entries on
    antialiasing greys: at 64 colours the orange and the red series came out the
    same red. 200 keeps every series colour and the file is still under 150 KB.
    """
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(colours, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], loop=0, duration=durations,
              optimize=True)
    return path


def camouflage_sweep(out: Path, fps: int = 12, per_stop: int = 15) -> Path:
    """Step the camouflage dial and watch the two quantities come apart.

    Nothing is interpolated. Each stop is one of the four measured camouflage
    levels, held for `per_stop` frames, and the right panel shows the 20
    individual runs at that stop rather than their mean.
    """
    runs = pd.read_csv(REPORTS / "detection_raw.csv").query("model == 'gcn'")
    means = runs.groupby("camouflage")[["auc", "ring_recall"]].mean()
    stops = means.index.to_numpy()

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.3))

    def draw(frame: int) -> None:
        stop = min(frame // per_stop, len(stops) - 1)
        step = frame % per_stop
        size = min(7.0, 3.0 + 1.4 * step)

        left.clear()
        left.plot(stops[:stop + 1], means.auc.to_numpy()[:stop + 1], "o-",
                  color=PALETTE[0], lw=2.2, label="node AUC")
        left.plot(stops[:stop + 1], means.ring_recall.to_numpy()[:stop + 1], "o-",
                  color=PALETTE[1], lw=2.2, label="ring recovery")
        left.axhline(0.5, color="#bbbbbb", lw=0.9, ls=":", zorder=0)
        left.set_xlim(-0.12, 2.12)
        left.set_ylim(-0.03, 1.05)
        left.set_xlabel(CAMO_AXIS, fontsize=9.5)
        left.set_ylabel("score (fraction, 0 to 1)")
        left.legend(loc="lower left")
        left.text(0.98, 0.97,
                  f"camouflage {stops[stop]:.1f}\n"
                  f"AUC {means.auc.iloc[stop]:.2f}\n"
                  f"rings {means.ring_recall.iloc[stop] * 100:.0f}%",
                  transform=left.transAxes, ha="right", va="top", fontsize=9.5,
                  color="#444444")
        titled(left, "Detection holds, ring recovery does not",
               "GCN, mean of 4 topologies x 5 seeds")

        right.clear()
        for earlier in range(stop):
            past = runs[runs.camouflage == stops[earlier]]
            right.plot(past.auc, past.ring_recall, "o", markersize=4,
                       color="#d3d3d3", zorder=1)
        now = runs[runs.camouflage == stops[stop]]
        for topology, group in now.groupby("topology"):
            right.plot(group.auc, group.ring_recall, "o", markersize=size,
                       color=TOPOLOGY_COLOURS[topology], label=topology, zorder=3)
        right.set_xlim(0.40, 1.03)
        right.set_ylim(-0.05, 1.06)
        right.set_xlabel("node AUC (0.5 = chance)", fontsize=9.5)
        right.set_ylabel("rings recovered (fraction)")
        right.legend(loc="upper left", title="ring topology", title_fontsize=8.5,
                     fontsize=8.5)
        titled(right, "One dot per run, not the mean",
               "20 runs per level; grey dots are the earlier levels")

        figure.tight_layout()

    anim = FuncAnimation(figure, draw, frames=len(stops) * per_stop, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return _shrink_gif(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        detection_vs_recovery(FIGURES / "detection-vs-recovery.png"),
        faithfulness(FIGURES / "faithfulness.png"),
        by_topology(FIGURES / "by-topology.png"),
        model_comparison(FIGURES / "model-comparison.png"),
        budget_sensitivity(FIGURES / "budget-sensitivity.png"),
        camouflage_sweep(FIGURES / "camouflage-sweep.gif"),
    ):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
