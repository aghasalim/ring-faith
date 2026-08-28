"""Build the README tables from reports/*.csv, so every number is a measured one.

    python experiments/make_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports")


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


CELL = ["topology", "camouflage", "seed", "model"]


def cell_means(frame: pd.DataFrame, value: str = "lift") -> pd.Series:
    """Mean of `value` per experimental cell.

    Individual nodes inside a cell share a graph and a trained model, so they are not
    independent. Anything compared across the detected/missed boundary is therefore paired at
    the cell level rather than pooled over nodes, which would overstate n by two orders of
    magnitude.
    """
    return frame.groupby(CELL)[value].mean()


def detected_vs_missed(fai_all: pd.DataFrame) -> pd.DataFrame:
    """Faithfulness on fraud nodes the model detected against ones it missed.

    The first version of this repo measured explanations only where the model was already
    right. A missed ring is the case an investigator most needs an explanation for, so this
    is the table that closes that gap. Both groups explain the fraud class, share the same
    analytic null, and are paired by cell.
    """
    oracle = fai_all[fai_all.k_mode == "oracle"]
    rows = []
    for (model, ex), sub in oracle.groupby(["model", "explainer"]):
        hit, miss = sub[sub.detected == 1], sub[sub.detected == 0]
        a, b = cell_means(hit), cell_means(miss)
        both = pd.concat([a, b], axis=1, join="inner", keys=["hit", "miss"]).dropna()
        stat = (
            wilcoxon(both["hit"], both["miss"]).pvalue
            if len(both) > 5 and not (both["hit"] - both["miss"] == 0).all()
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "explainer": ex,
                "detected lift": round(hit.lift.mean(), 3),
                "detected precision": round(hit.precision.mean(), 3),
                "detected n": len(hit),
                "missed lift": round(miss.lift.mean(), 3),
                "missed precision": round(miss.precision.mean(), 3),
                "missed n": len(miss),
                "lift difference": round(hit.lift.mean() - miss.lift.mean(), 3),
                "paired cells": len(both),
                "wilcoxon p": f"{stat:.2e}" if stat == stat else "-",
            }
        )
    return pd.DataFrame(rows)


def budget_sensitivity(fai_all: pd.DataFrame, detected: int) -> pd.DataFrame:
    """Mean lift over the null at each explanation budget.

    The oracle budget -- k set to the true number of motif edges in the candidate set -- is
    the one the headline claims were measured at, and it is information no investigator has.
    The `random` row is the control: the analytic null carries no k term, so a uniform
    ranking must sit at lift ~1.0 in every column, and any drift there is the measurement
    breaking rather than the explainers changing.
    """
    sub = fai_all[fai_all.detected == detected]
    piv = sub.pivot_table(index="explainer", columns="k_mode", values="lift", aggfunc="mean")
    order = [c for c in ("k1", "k3", "k5", "k10", "k20", "oracle") if c in piv.columns]
    return piv[order].round(3).reset_index()


def budget_head_to_head(fai_all: pd.DataFrame) -> pd.DataFrame:
    """Does the gradient still beat GNNExplainer once the budget stops being an oracle?

    Paired on the same node, the same model and the same budget. This is the test that can
    overturn finding F5, so it is reported at every budget rather than only where it wins.
    """
    rows = []
    for k_mode, sub in fai_all[fai_all.detected == 1].groupby("k_mode"):
        paired = sub.pivot_table(
            index=["topology", "camouflage", "seed", "model", "node"],
            columns="explainer",
            values="precision",
        ).dropna()
        for challenger in ("grad", "ig"):
            if challenger not in paired:
                continue
            d = paired[challenger] - paired["gnnexplainer"]
            p = wilcoxon(paired[challenger], paired["gnnexplainer"]).pvalue if d.any() else 1.0
            rows.append(
                {
                    "budget": k_mode,
                    "comparison": f"{challenger} vs gnnexplainer",
                    "challenger lift": round(sub[sub.explainer == challenger].lift.mean(), 3),
                    "gnnexpl lift": round(sub[sub.explainer == "gnnexplainer"].lift.mean(), 3),
                    "mean precision margin": round(d.mean(), 4),
                    "wins / ties / losses (%)": f"{100 * (d > 0).mean():.1f} / "
                    f"{100 * (d == 0).mean():.1f} / {100 * (d < 0).mean():.1f}",
                    "wilcoxon p": f"{p:.2e}",
                    "n": len(paired),
                }
            )
    df = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(["k1", "k3", "k5", "k10", "k20", "oracle"])}
    return df.sort_values(
        ["comparison", "budget"], key=lambda s: s.map(order) if s.name == "budget" else s
    ).reset_index(drop=True)


def main() -> None:
    det = pd.read_csv(OUT / "detection_raw.csv")
    fai_all = pd.read_csv(OUT / "faithfulness_raw.csv.gz")
    # Tables 1-6 are the originally published measurement: the oracle budget, on fraud nodes
    # the model detected. Everything added later is a separate table rather than a silent
    # change to those numbers.
    fai = fai_all[(fai_all.k_mode == "oracle") & (fai_all.detected == 1)]
    parts: list[str] = []

    # 1. Headline: detection vs faithfulness, per topology x camouflage (GCN).
    d = det[det.model == "gcn"].groupby(["topology", "camouflage"])[["auc", "ring_recall"]].mean()
    gsub = fai[(fai.model == "gcn") & (fai.explainer == "gnnexplainer")]
    f = gsub.groupby(["topology", "camouflage"])[
        ["precision", "random_expectation", "lift", "n_candidates"]
    ].mean()
    # Support matters: faithfulness is only measured on correctly-detected fraud
    # nodes, so a cell where detection collapses is backed by very few nodes.
    f["n_nodes_explained"] = gsub.groupby(["topology", "camouflage"]).size()
    h = d.join(f).reset_index().round(3)
    h.columns = [
        "topology", "camouflage", "node AUC", "ring recall",
        "GNNExpl precision", "random null", "lift over null", "candidate edges",
        "nodes explained",
    ]
    parts.append("### Table 1. GCN: detection vs explanation faithfulness\n\n" + md(h))

    # 2. Explainer comparison, pooled over topology and camouflage.
    e = (
        fai.groupby(["model", "explainer"])[["precision", "random_expectation", "lift"]]
        .agg(["mean", "std"])
        .round(3)
    )
    e.columns = [f"{a} {b}" for a, b in e.columns]
    parts.append("### Table 2. explainers, pooled over all cells\n\n" + md(e.reset_index()))

    # 3. Does structure help detection at all? (GNN vs structure-blind MLP)
    m = det.groupby(["topology", "model"])[["auc", "ring_recall"]].mean().round(3).reset_index()
    m = m.pivot(index="topology", columns="model", values=["auc", "ring_recall"])
    m.columns = [f"{a} ({b})" for a, b in m.columns]
    parts.append("### Table 3. structure-blind baseline\n\n" + md(m.reset_index()))

    # 4. The dissociation, isolated: rank topologies by AUC and by lift.
    diss = (
        h.groupby("topology")[["node AUC", "GNNExpl precision", "lift over null"]].mean().round(3)
    )
    diss["AUC rank"] = diss["node AUC"].rank(ascending=False).astype(int)
    diss["lift rank"] = diss["lift over null"].rank(ascending=False).astype(int)
    parts.append("### Table 4. the dissociation (averaged over camouflage)\n\n" + md(diss.reset_index()))

    # 5. Null calibration: the random explainer must sit at lift 1.0.
    r = fai[fai.explainer == "random"]
    cal = pd.DataFrame(
        [{
            "mean random precision": round(r.precision.mean(), 4),
            "mean analytic null": round(r.random_expectation.mean(), 4),
            "mean lift (should be ~1.0)": round(r.lift.mean(), 4),
            "n measurements": len(r),
        }]
    )
    parts.append("### Table 5. random-explainer control calibration\n\n" + md(cal))

    # 6. The statistics quoted in the README findings, so they are reproducible.
    stat_rows = []
    g16 = h.set_index(["topology", "camouflage"])
    for a, b in [
        ("node AUC", "GNNExpl precision"),
        ("node AUC", "lift over null"),
        ("random null", "GNNExpl precision"),
    ]:
        r, pv = pearsonr(g16[a], g16[b])
        stat_rows.append({"test": f"pearson r, {a} vs {b} (n=16 cells)", "stat": round(r, 3), "p": f"{pv:.2e}"})

    paired = fai.pivot_table(
        index=["topology", "camouflage", "seed", "model", "node"], columns="explainer", values="precision"
    ).dropna()
    d = paired["grad"] - paired["gnnexplainer"]
    stat_rows.append(
        {
            "test": f"wilcoxon, grad vs gnnexplainer precision (paired, n={len(paired)})",
            "stat": round(d.mean(), 4),
            "p": f"{wilcoxon(paired['grad'], paired['gnnexplainer']).pvalue:.2e}",
        }
    )
    stat_rows.append(
        {"test": "grad wins / ties / loses vs gnnexplainer (%)",
         "stat": f"{100 * (d > 0).mean():.1f} / {100 * (d == 0).mean():.1f} / {100 * (d < 0).mean():.1f}",
         "p": "-"}
    )
    for model in sorted(paired.index.get_level_values("model").unique()):
        q = paired.xs(model, level="model")
        for ex in ("gnnexplainer", "grad", "ig"):
            stat_rows.append(
                {
                    "test": f"{model} {ex}: beats own random null on % of nodes (n={len(q)})",
                    "stat": round(100 * (q[ex] > q["random"]).mean(), 1),
                    "p": f"{wilcoxon(q[ex], q['random']).pvalue:.2e}",
                }
            )
    for topo in sorted(h.topology.unique()):
        s = h[h.topology == topo].sort_values("camouflage")
        stat_rows.append(
            {
                "test": f"{topo}: spearman(camouflage, node AUC) / spearman(camouflage, lift)",
                "stat": f"{spearmanr(s.camouflage, s['node AUC']).statistic:+.1f} / "
                        f"{spearmanr(s.camouflage, s['lift over null']).statistic:+.1f}",
                "p": "-",
            }
        )
    parts.append("### Table 6. statistics quoted in the findings\n\n" + md(pd.DataFrame(stat_rows)))

    # 7. Explanations of fraud the model missed, against explanations of fraud it caught.
    parts.append(
        "### Table 7. faithfulness on detected vs missed fraud nodes (oracle budget)\n\n"
        + md(detected_vs_missed(fai_all))
    )

    # 8. What happens when the budget stops being an oracle.
    parts.append(
        "### Table 8. mean lift over the null by explanation budget (detected nodes)\n\n"
        + md(budget_sensitivity(fai_all, detected=1))
    )
    parts.append(
        "### Table 9. mean lift over the null by explanation budget (missed nodes)\n\n"
        + md(budget_sensitivity(fai_all, detected=0))
    )
    parts.append(
        "### Table 10. does the gradient still beat GNNExplainer at a realistic budget?\n\n"
        + md(budget_head_to_head(fai_all))
    )

    text = "\n\n".join(parts) + "\n"
    (OUT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
