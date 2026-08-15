"""Shows the id-ordering trap concretely, and what it does to reported AUC.

`generate` appends ring members after the background graph, so their node ids
are contiguous at the top of the range -- exactly what happens when you build a
graph in construction order. A contiguous split then puts every fraud node on
one side.

    python experiments/degenerate_split_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ringfaith.generate import generate
from ringfaith.metrics import node_metrics
from ringfaith.models import MODELS, fraud_scores, train
from ringfaith.split import contiguous_split, degenerate_classes, stratified_split


def main() -> None:
    torch.manual_seed(0)
    g = generate(topology="clique", camouflage=0.5, seed=0)
    x, e, y = torch.tensor(g.x), torch.tensor(g.edges), torch.tensor(g.y)
    out = {}

    for name, (tr, va, te) in {
        "contiguous": contiguous_split(g.n_nodes),
        "stratified": stratified_split(g.y, seed=0),
    }.items():
        bad = degenerate_classes(g.y, tr, va, te)
        row = {
            "train_fraud": int(g.y[tr].sum()),
            "val_fraud": int(g.y[va].sum()),
            "test_fraud": int(g.y[te].sum()),
            "degenerate_split_positions": bad,
        }
        if bad:
            row["test_auc"] = None
            row["note"] = "not trainable: a class is absent, so the AUC is undefined"
        else:
            m = train(MODELS["gcn"](g.x.shape[1]), x, e, y, tr, va)
            row["test_auc"] = round(node_metrics(g.y[te], fraud_scores(m, x, e).numpy()[te])["auc"], 4)
        out[name] = row
        print(f"{name:11s} {json.dumps(row)}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/degenerate_split.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/degenerate_split.json")


if __name__ == "__main__":
    main()
