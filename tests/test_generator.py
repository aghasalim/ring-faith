"""Tests for the instrument: does the generator produce what it claims?"""

import numpy as np
import pytest

from ringfaith.generate import TOPOLOGIES, generate

SMALL = dict(n_background=200, n_rings=4, ring_size=6, n_features=8)


def expected_motif_edges(topology, n_rings, s):
    per = {
        "clique": s * (s - 1) // 2,
        "star": s - 1,
        "cycle": s,
        "bipartite": (s // 2) * (s - s // 2),
    }[topology]
    return per * n_rings


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_motif_edge_count_matches_topology(topology):
    g = generate(topology=topology, seed=1, **SMALL)
    assert len(g.ring_edges) == expected_motif_edges(topology, SMALL["n_rings"], SMALL["ring_size"])


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_ring_edges_are_a_subset_of_edges(topology):
    """Edge-level ground truth must actually exist in the graph."""
    g = generate(topology=topology, camouflage=0.7, seed=2, **SMALL)
    mask = g.ring_edge_mask()
    assert mask.sum() == len(g.ring_edges)
    present = {tuple(e) for e in g.edges}
    assert all(tuple(e) in present for e in g.ring_edges)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_edges_are_canonical_unique_and_loop_free(topology):
    g = generate(topology=topology, camouflage=1.0, seed=3, **SMALL)
    assert (g.edges[:, 0] < g.edges[:, 1]).all(), "edges must be canonical u < v"
    assert len({tuple(e) for e in g.edges}) == len(g.edges), "duplicate edges"
    assert g.edges.max() < g.n_nodes


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_motif_edges_only_join_ring_members_of_the_same_ring(topology):
    g = generate(topology=topology, seed=4, **SMALL)
    u, v = g.ring_edges[:, 0], g.ring_edges[:, 1]
    assert (g.y[u] == 1).all() and (g.y[v] == 1).all()
    assert (g.ring_id[u] == g.ring_id[v]).all(), "rings must be disjoint"


def test_labels_are_exactly_ring_membership():
    g = generate(topology="clique", seed=5, **SMALL)
    assert g.y.sum() == SMALL["n_rings"] * SMALL["ring_size"]
    assert np.array_equal(g.y, (g.ring_id >= 0).astype(np.int64))
    for r in range(SMALL["n_rings"]):
        assert (g.ring_id == r).sum() == SMALL["ring_size"]


def test_camouflage_adds_edges_without_touching_ground_truth():
    """Camouflage must change only cover traffic, never the planted motif."""
    graphs = [generate(topology="clique", camouflage=c, seed=6, **SMALL) for c in (0.0, 0.5, 1.5)]
    counts = [g.n_edges for g in graphs]
    assert counts[0] < counts[1] < counts[2], f"camouflage did not add edges: {counts}"
    for g in graphs[1:]:
        assert np.array_equal(g.ring_edges, graphs[0].ring_edges)
        assert np.array_equal(g.y, graphs[0].y)


def test_camouflage_edges_attach_ring_members_to_legitimate_nodes():
    g = generate(topology="star", camouflage=1.0, seed=7, **SMALL)
    motif = {tuple(e) for e in g.ring_edges}
    touching_ring = [tuple(e) for e in g.edges if (g.y[e[0]] or g.y[e[1]]) and tuple(e) not in motif]
    assert touching_ring, "expected camouflage edges to exist"
    # Every non-motif edge on a ring node must lead to a legitimate node.
    assert all(g.y[u] + g.y[v] == 1 for u, v in touching_ring)


def test_generation_is_deterministic_in_the_seed():
    a, b = (generate(topology="cycle", camouflage=0.5, seed=11, **SMALL) for _ in range(2))
    c = generate(topology="cycle", camouflage=0.5, seed=12, **SMALL)
    assert np.array_equal(a.edges, b.edges) and np.allclose(a.x, b.x)
    assert not np.array_equal(a.edges, c.edges)


def test_background_degree_distribution_is_heavy_tailed():
    """Preferential attachment, not a uniform random graph."""
    g = generate(topology="clique", camouflage=0.0, n_background=1500, n_rings=1, ring_size=4, n_features=4, seed=13)
    deg = np.bincount(g.edges[:, :2][g.y[g.edges].sum(1) == 0].ravel(), minlength=g.n_nodes)[: 1500]
    deg = deg[deg > 0]
    assert deg.max() > 8 * deg.mean(), f"max degree {deg.max()} vs mean {deg.mean():.1f} is not scale-free-ish"


def test_ring_members_get_the_feature_shift():
    g = generate(topology="clique", feature_signal=5.0, seed=14, **SMALL)
    assert g.x[g.y == 1, 0].mean() - g.x[g.y == 0, 0].mean() > 3.0
    # Other dimensions carry no signal.
    assert abs(g.x[g.y == 1, 1].mean() - g.x[g.y == 0, 1].mean()) < 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(topology="triangle"),
        dict(camouflage=-0.1),
        dict(n_background=2, m_attach=5),
        dict(topology="bipartite", ring_size=1),
    ],
)
def test_invalid_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        generate(**{**SMALL, **kwargs})
