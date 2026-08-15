"""Tests for the measurement layer: splits, metrics, and the random null.

The random-explainer control is the load-bearing part of this repo, so it gets
tested against its closed form rather than eyeballed.
"""

import numpy as np
import pytest
import torch

from ringfaith import explain
from ringfaith.generate import generate
from ringfaith.metrics import edge_faithfulness, node_metrics, ring_recall
from ringfaith.models import MODELS, dense_adj
from ringfaith.split import (
    assert_not_degenerate,
    contiguous_split,
    degenerate_classes,
    stratified_split,
)

SMALL = dict(n_background=200, n_rings=4, ring_size=6, n_features=8)


# --- splits -----------------------------------------------------------------

def test_contiguous_split_is_degenerate_and_stratified_is_not():
    """The id-ordering trap: ring members are appended, so contiguous splitting
    puts every fraud node in the test set and none in train."""
    g = generate(topology="clique", seed=0, **SMALL)
    assert degenerate_classes(g.y, *contiguous_split(g.n_nodes)) == [0, 1]
    assert g.y[contiguous_split(g.n_nodes)[0]].sum() == 0
    tr, va, te = stratified_split(g.y, seed=0)
    assert degenerate_classes(g.y, tr, va, te) == []
    assert_not_degenerate(g.y, tr, va, te)


def test_assert_not_degenerate_actually_raises():
    y = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="degenerate"):
        assert_not_degenerate(y, np.array([0, 1, 2]), np.array([3, 4, 5]))


def test_stratified_split_is_a_disjoint_partition_with_both_classes():
    g = generate(topology="star", seed=1, **SMALL)
    tr, va, te = stratified_split(g.y, seed=3)
    assert sorted(np.concatenate([tr, va, te])) == list(range(g.n_nodes))
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    for s in (tr, va, te):
        assert 0 < g.y[s].sum() < len(s)


# --- node / ring metrics ----------------------------------------------------

def test_node_metrics_hit_the_endpoints():
    y = np.array([0, 0, 1, 1])
    assert node_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auc"] == 1.0
    assert node_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["auc"] == 0.0
    assert np.isnan(node_metrics(np.zeros(4, dtype=int), np.arange(4.0))["auc"])


def test_ring_recall_endpoints_and_threshold():
    g = generate(topology="clique", seed=2, **SMALL)
    assert ring_recall(g.y, g.ring_id, g.y.astype(float)) == 1.0
    assert ring_recall(g.y, g.ring_id, 1.0 - g.y.astype(float)) == 0.0
    # Flag one whole ring and nothing else -> exactly 1 of n_rings recovered.
    s = (g.ring_id == 0).astype(float)
    assert ring_recall(g.y, g.ring_id, s) == pytest.approx(1 / SMALL["n_rings"])


# --- faithfulness and the null ----------------------------------------------

def test_faithfulness_endpoints():
    cand = np.arange(20)
    mask = np.zeros(20, dtype=bool)
    mask[:5] = True
    oracle = np.where(mask, 1.0, 0.0)
    r = edge_faithfulness(cand, oracle, mask)
    assert r["precision"] == r["recall"] == r["f1"] == 1.0
    assert r["k"] == r["n_relevant"] == 5 and r["n_candidates"] == 20
    assert r["random_expectation"] == pytest.approx(0.25) and r["lift"] == pytest.approx(4.0)
    assert edge_faithfulness(cand, -oracle, mask)["precision"] == 0.0


def test_faithfulness_is_none_when_no_motif_edge_is_reachable():
    assert edge_faithfulness(np.arange(10), np.random.rand(10), np.zeros(10, dtype=bool)) is None


def test_random_explainer_matches_its_analytic_null():
    """E[precision@k] with k = n_relevant must equal n_relevant / n_candidates."""
    n_cand, n_rel, trials = 60, 12, 4000
    mask = np.zeros(n_cand, dtype=bool)
    mask[:n_rel] = True
    rng = np.random.default_rng(0)
    p = [
        edge_faithfulness(np.arange(n_cand), rng.random(n_cand), mask, seed=t)["precision"]
        for t in range(trials)
    ]
    assert np.mean(p) == pytest.approx(n_rel / n_cand, abs=0.02)


def test_all_tied_scores_are_broken_without_index_order_bias():
    """Ring edges sit at the END of the edge array (highest node ids). With
    index-order tie-breaking a constant score would score 0 precision; random
    tie-breaking must land on the null instead."""
    n_cand, n_rel = 60, 12
    mask = np.zeros(n_cand, dtype=bool)
    mask[-n_rel:] = True  # relevant edges last, as in a real generated graph
    tied = np.ones(n_cand)
    p = [edge_faithfulness(np.arange(n_cand), tied, mask, seed=t)["precision"] for t in range(4000)]
    assert np.mean(p) == pytest.approx(n_rel / n_cand, abs=0.02)
    assert min(p) < max(p), "tie-breaking is deterministic, so it is biased"


# --- model / explainer plumbing ---------------------------------------------

def test_dense_adj_is_symmetric_and_differentiable_in_the_edge_weight():
    edges = torch.tensor([[0, 1], [1, 2]])
    w = torch.ones(2, requires_grad=True)
    a = dense_adj(edges, 3, w)
    assert torch.equal(a, a.T)
    assert torch.equal(a, dense_adj(edges, 3, None))
    a.sum().backward()
    assert torch.allclose(w.grad, torch.full((2,), 2.0)), "each undirected edge fills two cells"


@pytest.mark.parametrize("name", ["gcn", "sage"])
def test_unit_edge_weights_reproduce_the_unweighted_forward(name):
    g = generate(topology="cycle", seed=4, **SMALL)
    x, e = torch.tensor(g.x), torch.tensor(g.edges)
    m = MODELS[name](g.x.shape[1])
    m.train(False)
    with torch.no_grad():
        assert torch.allclose(m(x, e), m(x, e, torch.ones(len(g.edges))), atol=1e-5)


def test_candidate_set_is_the_khop_subgraph_and_holds_the_targets_own_edges():
    g = generate(topology="clique", camouflage=0.5, seed=5, **SMALL)
    adj = explain.adjacency_list(g.edges, g.n_nodes)
    target = int(np.flatnonzero(g.y == 1)[0])
    cand = explain.candidate_edges(g.edges, adj, g.n_nodes, target, hops=2)
    incident = set(np.flatnonzero((g.edges == target).any(1)).tolist())
    assert incident <= set(cand.tolist()), "1-hop edges must be candidates"
    inside = set(explain.khop_nodes(adj, target, 2).tolist())
    assert all(u in inside and v in inside for u, v in g.edges[cand])
    assert g.ring_edge_mask()[cand].sum() > 0, "a ring member must reach its own motif edges"


def test_every_explainer_scores_the_same_candidate_set():
    """Without this the random control is not a valid null."""
    g = generate(topology="bipartite", camouflage=0.5, seed=6, **SMALL)
    x, e = torch.tensor(g.x), torch.tensor(g.edges)
    m = MODELS["gcn"](g.x.shape[1])
    m.train(False)
    adj = explain.adjacency_list(g.edges, g.n_nodes)
    target = int(np.flatnonzero(g.y == 1)[0])
    cand = explain.candidate_edges(g.edges, adj, g.n_nodes, target, hops=2)
    for name, fn in explain.REGISTRY.items():
        s = fn(m, x, e, target, cand, epochs=5) if name == "gnnexplainer" else fn(m, x, e, target, cand)
        assert len(s) == len(cand), name
        assert np.isfinite(s).all(), name
