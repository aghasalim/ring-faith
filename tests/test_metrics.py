"""Tests for the measurement layer: splits, metrics, and the random null.

The random-explainer control is the load-bearing part of this repo, so it gets
tested against its closed form rather than eyeballed.
"""

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("k", [1, 3, 5, 10, 30])
def test_the_random_null_holds_at_every_explanation_budget(k):
    """The k sweep is only readable if lift means the same thing at every budget.

    E[precision@k] for a uniform ranking is `n_relevant / n_candidates` for *any* k, not just
    the oracle k = n_relevant, so the analytic null needs no k term. This is the control for
    the budget-sensitivity measurement, and it is the reason a lift at k=3 can be compared
    with a lift at the oracle budget.
    """
    n_cand, n_rel, trials = 60, 12, 4000
    mask = np.zeros(n_cand, dtype=bool)
    mask[:n_rel] = True
    rng = np.random.default_rng(1)
    p = [
        edge_faithfulness(np.arange(n_cand), rng.random(n_cand), mask, k=k, seed=t)["precision"]
        for t in range(trials)
    ]
    assert np.mean(p) == pytest.approx(n_rel / n_cand, abs=0.02)


def test_budget_larger_than_the_candidate_set_is_clamped():
    """Otherwise precision would be divided by a k the ranking cannot fill, and a large
    budget would look artificially unfaithful."""
    n_cand, n_rel = 20, 5
    mask = np.zeros(n_cand, dtype=bool)
    mask[:n_rel] = True
    r = edge_faithfulness(np.arange(n_cand), np.random.default_rng(0).random(n_cand), mask, k=999)
    assert r["k"] == n_cand
    assert r["precision"] == pytest.approx(n_rel / n_cand)
    assert r["recall"] == 1.0


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


def test_integrated_gradients_is_the_path_averaged_gradient():
    """One midpoint step must be exactly the plain gradient read at edge weight 0.5.

    That pins the contract that separates `ig` from `grad`: the same quantity, averaged along
    the path from the empty graph to the real one rather than sampled at unit weights.
    """
    g = generate(topology="star", camouflage=0.5, seed=8, **SMALL)
    x, e = torch.tensor(g.x), torch.tensor(g.edges)
    m = MODELS["gcn"](g.x.shape[1])
    m.train(False)
    adj = explain.adjacency_list(g.edges, g.n_nodes)
    target = int(np.flatnonzero(g.y == 1)[0])
    cand = explain.candidate_edges(g.edges, adj, g.n_nodes, target, hops=2)

    w = torch.full((len(g.edges),), 0.5, requires_grad=True)
    ref = torch.autograd.grad(m(x, e, w)[target, explain.FRAUD], w)[0].abs().numpy()[cand]
    got = explain.ig(m, x, e, target, cand, target_class=explain.FRAUD, steps=1)
    assert np.allclose(ref, got, atol=1e-6)


@pytest.mark.parametrize("name", ["gnnexplainer", "grad", "ig"])
def test_target_class_overrides_the_predicted_class(name):
    """Faithfulness on missed fraud nodes is only comparable with faithfulness on detected
    ones if both explain the same class. On a node the model calls legitimate, the default
    (explain the prediction) and `target_class=FRAUD` must genuinely differ; on a node it
    calls fraud they must be identical, which is why the published numbers do not move.
    """
    g = generate(topology="cycle", camouflage=1.0, seed=9, **SMALL)
    x, e = torch.tensor(g.x), torch.tensor(g.edges)
    torch.manual_seed(0)  # the model is untrained, so pin it rather than inherit RNG order
    m = MODELS["gcn"](g.x.shape[1])
    m.train(False)
    adj = explain.adjacency_list(g.edges, g.n_nodes)
    node = int(np.flatnonzero(g.y == 1)[0])
    with torch.no_grad():
        predicted = int(m(x, e)[node].argmax())
    cand = explain.candidate_edges(g.edges, adj, g.n_nodes, node, hops=2)
    kw = dict(epochs=5) if name == "gnnexplainer" else {}

    default = explain.REGISTRY[name](m, x, e, node, cand, **kw)
    same = explain.REGISTRY[name](m, x, e, node, cand, target_class=predicted, **kw)
    other = explain.REGISTRY[name](m, x, e, node, cand, target_class=1 - predicted, **kw)
    assert np.allclose(default, same), "naming the predicted class must change nothing"
    assert not np.allclose(default, other), "naming the other class must change something"


def test_run_config_measures_missed_nodes_at_every_budget():
    """The gap this closes: faithfulness used to be defined only where the model was right."""
    from ringfaith.experiment import K_MODES, run_config

    _, faith = run_config(
        "clique",
        2.0,
        seed=0,
        models=("gcn",),
        explainers=("grad", "random"),
        n_explain=3,
        gen_kwargs=SMALL,
    )
    rows = pd.DataFrame(faith)
    assert set(rows.k_mode) == set(K_MODES), "every budget must be evaluated"
    assert set(rows.detected) == {0, 1}, "both detected and missed fraud nodes must appear"
    # the budget actually applied is the budget asked for, clamped to the candidate set
    for _, r in rows[rows.k_mode == "k3"].iterrows():
        assert r["k"] == min(3, r["n_candidates"])


def test_mean_aggregation_makes_integrated_gradients_equal_to_the_plain_gradient():
    """Why `ig` and `grad` produce identical numbers on GraphSAGE, and only there.

    SAGE's mean aggregation divides by the row sum of the weighted adjacency, so scaling
    every edge weight by the same alpha cancels: the forward pass is invariant along the
    straight-line path integrated gradients walks. A degree-zero homogeneous function has
    grad(alpha*w) = grad(w)/alpha, so the path average is a positive multiple of the gradient
    at unit weights and the two rank the candidate edges identically. GCN's normalisation
    adds the identity, which breaks the homogeneity, and there they differ.
    """
    g = generate(topology="clique", camouflage=0.5, seed=11, **SMALL)
    x, e = torch.tensor(g.x), torch.tensor(g.edges)
    adj = explain.adjacency_list(g.edges, g.n_nodes)
    target = int(np.flatnonzero(g.y == 1)[0])
    cand = explain.candidate_edges(g.edges, adj, g.n_nodes, target, hops=2)

    sage = MODELS["sage"](g.x.shape[1])
    sage.train(False)
    with torch.no_grad():
        half = sage(x, e, torch.full((len(g.edges),), 0.5))
        one = sage(x, e, torch.ones(len(g.edges)))
    assert torch.allclose(half, one, atol=1e-5), "mean aggregation must be scale invariant"

    a = explain.grad(sage, x, e, target, cand, target_class=explain.FRAUD)
    b = explain.ig(sage, x, e, target, cand, target_class=explain.FRAUD, steps=8)
    assert np.array_equal(np.argsort(-a), np.argsort(-b)), "same ranking on SAGE"

    gcn = MODELS["gcn"](g.x.shape[1])
    gcn.train(False)
    a = explain.grad(gcn, x, e, target, cand, target_class=explain.FRAUD)
    b = explain.ig(gcn, x, e, target, cand, target_class=explain.FRAUD, steps=8)
    assert not np.array_equal(np.argsort(-a), np.argsort(-b)), "GCN is not scale invariant"
