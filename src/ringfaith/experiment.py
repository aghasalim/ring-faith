"""One experimental cell: generate -> split -> train -> score -> explain."""

from __future__ import annotations

import numpy as np
import torch

from ringfaith import explain
from ringfaith.generate import generate
from ringfaith.metrics import edge_faithfulness, node_metrics, ring_recall
from ringfaith.models import MODELS, fraud_scores, train
from ringfaith.split import assert_not_degenerate, stratified_split

# The explanation budget. `None` is the oracle budget -- the true number of motif edges in
# the candidate set -- which is what the original sweep used and is generous to every
# explainer, since an investigator does not know that number. The fixed budgets are what a
# review queue actually looks like: show me the top few edges. The analytic random null is
# `n_relevant / n_candidates` regardless of k, so lift stays well defined at every budget.
K_MODES: dict[str, int | None] = {"oracle": None, "k1": 1, "k3": 3, "k5": 5, "k10": 10, "k20": 20}


def run_config(
    topology: str,
    camouflage: float,
    seed: int,
    models: tuple[str, ...] = ("gcn", "sage", "mlp"),
    explainers: tuple[str, ...] = explain.EXPLAINERS,
    n_explain: int = 25,
    hops: int = 2,
    gen_kwargs: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run one (topology, camouflage, seed) cell.

    Returns (detection_rows, faithfulness_rows). Faithfulness is measured only for graph
    models -- the MLP has no edges to explain -- on fraud nodes the model detected
    (`score > 0.5`) and, separately, on fraud nodes it missed. The missed group is the
    operationally interesting one and was not measured at all in the first version of this
    repo; rows carry `detected` so the two can be compared or either read alone.

    Both groups are explained with respect to the fraud class, not the predicted class, so
    they answer the same question. Every explainer's scores are evaluated at every budget in
    `K_MODES` from a single scoring pass, so the budget sweep costs nothing extra.
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

        # Two groups, sampled reproducibly and from independent streams so that adding the
        # missed group cannot perturb which detected nodes get drawn.
        groups = {
            1: (np.flatnonzero((g.y == 1) & (scores > 0.5)), 1000 + seed),
            0: (np.flatnonzero((g.y == 1) & (scores <= 0.5)), 2000 + seed),
        }
        for detected, (pool, draw_seed) in groups.items():
            if len(pool) == 0:
                continue
            targets = np.random.default_rng(draw_seed).permutation(pool)[:n_explain]
            for target in targets:
                cand = explain.candidate_edges(g.edges, adj, g.n_nodes, int(target), hops=hops)
                if ring_mask[cand].sum() == 0:
                    continue
                for ex in explainers:
                    s = explain.REGISTRY[ex](
                        model, x, e, int(target), cand, seed=seed, target_class=explain.FRAUD
                    )
                    for k_mode, k in K_MODES.items():
                        r = edge_faithfulness(
                            cand, s, ring_mask, k=k, seed=seed * 100003 + int(target)
                        )
                        if r is not None:
                            faith_rows.append(
                                {
                                    **base,
                                    "model": name,
                                    "explainer": ex,
                                    "node": int(target),
                                    "detected": detected,
                                    "k_mode": k_mode,
                                    **r,
                                }
                            )
    return det_rows, faith_rows
