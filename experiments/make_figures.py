"""Draw the README figures from reports/*.csv.

Reads the saved sweep output only, so this is free to run and the pictures cannot
drift from the tables.  Both figures exist to separate quantities the field tends
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
EXPLAINER_COLOURS = {
    "ig": "#2166ac",
    "grad": "#67a9cf",
    "gnnexplainer": "#92c5de",
    "random": "#bdbdbd",
}


def _load(name: str, columns: list[str]) -> pd.DataFrame:
    """Read one of the two-row-header summaries and give it flat column names."""
    table = pd.read_csv(REPORTS / name, header=[0, 1], skiprows=[2])
    table.columns = columns
    return table


def detection_vs_recovery(out: Path) -> Path:
    """Show AUC holding up while ring recovery collapses.

    Camouflage is the x-axis because it is the only knob that separates the two:
    a model can keep ranking fraudulent nodes above clean ones while finding none
    of the rings that make the fraud worth investigating.
    """
    table = _load("detection_summary.csv", DETECTION_COLUMNS)
    gcn = table[table.model == "gcn"]
    grouped = gcn.groupby("camouflage")[["auc", "ap", "ring_recall"]].mean()
    x = grouped.index.to_numpy()

    figure, ax = plt.subplots(figsize=(8.2, 4.8))
    for column, label, colour in [
        ("auc", "node AUC", "#2166ac"),
        ("ap", "average precision", "#f4a582"),
        ("ring_recall", "ring recovery", "#b2182b"),
    ]:
        ax.plot(x, grouped[column], "o-", color=colour, label=label, lw=2)
    ax.axhline(0.5, color="0.7", lw=0.9, ls=":")
    ax.annotate(
        f"AUC still {grouped.auc.iloc[-1]:.2f}\nrings found: "
        f"{grouped.ring_recall.iloc[-1] * 100:.0f}%",
        xy=(x[-1], grouped.ring_recall.iloc[-1]),
        xytext=(x[-1] - 0.75, 0.42),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    ax.set_xlabel("camouflage")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "GCN, averaged over topologies and seeds.\n"
        "The headline metric degrades gently while the thing an investigator "
        "needs goes to nothing.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def faithfulness(out: Path) -> Path:
    """Explainer precision against the random-edge expectation it must be read against.

    The random explainer is the control: it should land on a lift of exactly 1, and
    any deviation would mean the null is mis-specified rather than that random
    edge selection works.
    """
    table = _load("faithfulness_summary.csv", FAITH_COLUMNS).dropna(subset=["lift"])
    detected = table[
        (table.model == "gcn") & (table.detected.astype(str) == "1")
    ]
    explainers = ["ig", "grad", "gnnexplainer", "random"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))

    summary = detected.groupby("explainer")[["precision", "random_expectation"]].mean()
    summary = summary.loc[explainers]
    base = np.arange(len(explainers))
    left.bar(base - 0.2, summary.precision, 0.4, label="explainer precision",
             color=[EXPLAINER_COLOURS[e] for e in explainers], edgecolor="0.3", lw=0.5)
    left.bar(base + 0.2, summary.random_expectation, 0.4, label="random-edge expectation",
             color="none", edgecolor="0.35", lw=1.2, hatch="///")
    left.set_xticks(base)
    left.set_xticklabels(explainers)
    left.set_ylabel("fraction of top edges inside the ring")
    left.set_title("at best 41% of the explained edges are ring edges", fontsize=10)
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    by_camouflage = (
        detected[detected.explainer != "random"]
        .groupby(["camouflage", "explainer"])["lift"].mean().unstack()
    )
    for explainer in ["ig", "grad", "gnnexplainer"]:
        right.plot(by_camouflage.index, by_camouflage[explainer], "o-",
                   color=EXPLAINER_COLOURS[explainer], label=explainer, lw=2)
    control = detected[detected.explainer == "random"].groupby("camouflage")["lift"].mean()
    right.plot(control.index, control, "s--", color="0.55", label="random (control)", lw=1.4)
    right.axhline(1.0, color="0.2", lw=1.0)
    right.set_xlabel("camouflage")
    right.set_ylabel("lift over the random-edge null")
    right.set_title(
        "lift rises as camouflage thins the null; the control stays on 1", fontsize=10
    )
    right.legend(frameon=False, fontsize=8)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        detection_vs_recovery(FIGURES / "detection-vs-recovery.png"),
        faithfulness(FIGURES / "faithfulness.png"),
    ):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
