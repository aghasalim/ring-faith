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


def main() -> None:
    det = pd.read_csv(OUT / "detection_raw.csv")
    fai = pd.read_csv(OUT / "faithfulness_raw.csv")
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
    parts.append("### Table 1 — GCN: detection vs explanation faithfulness\n\n" + md(h))

    # 2. Explainer comparison, pooled over topology and camouflage.
    e = (
        fai.groupby(["model", "explainer"])[["precision", "random_expectation", "lift"]]
        .agg(["mean", "std"])
        .round(3)
    )
    e.columns = [f"{a} {b}" for a, b in e.columns]
    parts.append("### Table 2 — explainers, pooled over all cells\n\n" + md(e.reset_index()))

    # 3. Does structure help detection at all? (GNN vs structure-blind MLP)
    m = det.groupby(["topology", "model"])[["auc", "ring_recall"]].mean().round(3).reset_index()
    m = m.pivot(index="topology", columns="model", values=["auc", "ring_recall"])
    m.columns = [f"{a} ({b})" for a, b in m.columns]
    parts.append("### Table 3 — structure-blind baseline\n\n" + md(m.reset_index()))

    # 4. The dissociation, isolated: rank topologies by AUC and by lift.
    diss = (
        h.groupby("topology")[["node AUC", "GNNExpl precision", "lift over null"]].mean().round(3)
    )
    diss["AUC rank"] = diss["node AUC"].rank(ascending=False).astype(int)
    diss["lift rank"] = diss["lift over null"].rank(ascending=False).astype(int)
    parts.append("### Table 4 — the dissociation (averaged over camouflage)\n\n" + md(diss.reset_index()))

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
    parts.append("### Table 5 — random-explainer control calibration\n\n" + md(cal))

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
        for ex in ("gnnexplainer", "grad"):
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
    parts.append("### Table 6 — statistics quoted in the findings\n\n" + md(pd.DataFrame(stat_rows)))

    text = "\n\n".join(parts) + "\n"
    (OUT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
