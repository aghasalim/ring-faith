"""One experimental cell: generate -> split -> train -> score -> explain."""

from __future__ import annotations

import numpy as np
import torch

from ringfaith import explain
from ringfaith.generate import generate
from ringfaith.metrics import edge_faithfulness, node_metrics, ring_recall
from ringfaith.models import MODELS, fraud_scores, train
from ringfaith.split import assert_not_degenerate, stratified_split


def run_config(
    topology: str,
    camouflage: float,
    seed: int,
    models: tuple[str, ...] = ("gcn", "sage", "mlp"),
    explainers: tuple[str, ...] = ("gnnexplainer", "grad", "random"),
    n_explain: int = 25,
    hops: int = 2,
    gen_kwargs: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run one (topology, camouflage, seed) cell.

    Returns (detection_rows, faithfulness_rows). Faithfulness is measured only
    for graph models -- the MLP has no edges to explain -- and only on fraud
    nodes the model actually got right, since explaining a missed node asks a
    different question.
    """
    torch.manual_seed(seed)
    g = generate(topology=topology, camouflage=camouflage, seed=seed, **(gen_kwargs or {}))
    x, e, y = torch.tensor(g.x), torch.tensor(g.edges), torch.tensor(g.y)
    tr, va, te = stratified_split(g.y, seed=seed)
    assert_not_degenerate(g.y, tr, va, te)

    adj = explain.adjacency_list(g.edges, g.n_nodes)
    ring_mask = g.ring_edge_mask()
    base = dict(
        topology=topology,
        camouflage=camouflage,
        seed=seed,
        n_nodes=g.n_nodes,
        n_edges=g.n_edges,
        n_ring_edges=len(g.ring_edges),
    )
    det_rows, faith_rows = [], []

    for name in models:
        model = train(MODELS[name](g.x.shape[1]), x, e, y, tr, va)
        scores = fraud_scores(model, x, e).numpy()
        det_rows.append(
            {
                **base,
                "model": name,
                **node_metrics(g.y[te], scores[te]),
                "ring_recall": ring_recall(g.y, g.ring_id, scores),
            }
        )
        if name == "mlp":
            continue

        # Explain correctly-detected fraud nodes, sampled reproducibly.
        rng = np.random.default_rng(1000 + seed)
        fraud = np.flatnonzero((g.y == 1) & (scores > 0.5))
        if len(fraud) == 0:
            continue
        targets = rng.permutation(fraud)[:n_explain]

        for target in targets:
            cand = explain.candidate_edges(g.edges, adj, g.n_nodes, int(target), hops=hops)
            if ring_mask[cand].sum() == 0:
                continue
            for ex in explainers:
                s = explain.REGISTRY[ex](model, x, e, int(target), cand, seed=seed)
                r = edge_faithfulness(cand, s, ring_mask, seed=seed * 100003 + int(target))
                if r is not None:
                    faith_rows.append({**base, "model": name, "explainer": ex, "node": int(target), **r})
    return det_rows, faith_rows
