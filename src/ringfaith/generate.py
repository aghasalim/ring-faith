"""Synthetic fraud graphs with planted collusion rings and exact ground truth.

Construction order (this order matters, see `README.md` finding F4):
  1. Barabasi-Albert background graph of `n_background` legitimate nodes.
  2. Ring member nodes are *appended* after the background, so their ids are
     contiguous at the end of the id range. This is deliberate: it reproduces
     the id-ordering hazard that makes a contiguous train/test split degenerate.
     Always split with `ringfaith.split.stratified_split`.
  3. The motif edges for each ring are added among its members.
  4. Camouflage edges are added from each ring member to *legitimate* nodes,
     chosen by preferential attachment, so a fraudster hides inside a normal
     looking neighbourhood.

Ground truth is exact: `y` (node labels), `ring_id` (which ring), and
`ring_edges` (the planted motif's own edges, a subset of `edges`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOPOLOGIES = ("clique", "star", "cycle", "bipartite")


@dataclass
class FraudGraph:
    """A generated graph plus its exact ground truth."""

    edges: np.ndarray  # [E, 2] int64, undirected, u < v, unique, no self loops
    x: np.ndarray  # [N, F] float32 node features
    y: np.ndarray  # [N] int64, 1 = ring member
    ring_id: np.ndarray  # [N] int64, ring index, -1 for legitimate
    ring_edges: np.ndarray  # [M, 2] int64, the planted motif edges
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    def ring_edge_mask(self) -> np.ndarray:
        """[E] bool, True where `edges[i]` is one of the planted motif edges."""
        if len(self.ring_edges) == 0:
            return np.zeros(self.n_edges, dtype=bool)
        key = self.edges[:, 0].astype(np.int64) * self.n_nodes + self.edges[:, 1]
        rkey = self.ring_edges[:, 0].astype(np.int64) * self.n_nodes + self.ring_edges[:, 1]
        return np.isin(key, rkey)


def _sample_distinct(pool: list[int], k: int, rng: np.random.Generator) -> list[int]:
    """Draw k distinct values from `pool` (a degree-repeated node list)."""
    out: set[int] = set()
    pool_arr = np.asarray(pool)
    # Rejection sampling; the pool is degree-weighted so this is preferential
    # attachment. Falls back to a uniform draw if the pool lacks k distinct ids.
    for _ in range(200 * k):
        if len(out) >= k:
            break
        out.add(int(pool_arr[rng.integers(len(pool_arr))]))
    if len(out) < k:
        extra = [v for v in np.unique(pool_arr) if v not in out]
        rng.shuffle(extra)
        out.update(int(v) for v in extra[: k - len(out)])
    return list(out)


def _barabasi_albert(n: int, m: int, rng: np.random.Generator):
    """Barabasi-Albert preferential attachment. Returns (edges, degree_pool)."""
    if n <= m:
        raise ValueError(f"need n > m, got n={n}, m={m}")
    edges: list[tuple[int, int]] = []
    targets = list(range(m))
    pool: list[int] = []
    for v in range(m, n):
        for u in targets:
            edges.append((u, v))
        pool.extend(targets)
        pool.extend([v] * m)
        targets = _sample_distinct(pool, m, rng)
    return edges, pool


def _motif_edges(members: list[int], topology: str) -> list[tuple[int, int]]:
    """The planted motif's own edges for one ring."""
    s = len(members)
    if topology == "clique":
        return [(members[i], members[j]) for i in range(s) for j in range(i + 1, s)]
    if topology == "star":
        return [(members[0], members[i]) for i in range(1, s)]
    if topology == "cycle":
        return [(members[i], members[(i + 1) % s]) for i in range(s)]
    if topology == "bipartite":
        half = s // 2
        mules, merchants = members[:half], members[half:]
        return [(a, b) for a in mules for b in merchants]
    raise ValueError(f"unknown topology {topology!r}, expected one of {TOPOLOGIES}")


def generate(
    n_background: int = 800,
    m_attach: int = 2,
    n_rings: int = 6,
    ring_size: int = 8,
    topology: str = "clique",
    camouflage: float = 0.0,
    n_features: int = 16,
    feature_signal: float = 0.35,
    seed: int = 0,
) -> FraudGraph:
    """Generate one fraud graph.

    Args:
        n_background: legitimate nodes in the Barabasi-Albert background.
        m_attach: BA attachment parameter (edges added per new background node).
        n_rings: number of disjoint collusion rings planted.
        ring_size: members per ring.
        topology: one of `TOPOLOGIES`.
        camouflage: each ring member adds `1 + round(camouflage * d_ring)` edges
            to legitimate nodes, where `d_ring` is its degree inside the motif.
            0.0 still gives one legitimate edge per member so the graph stays
            connected; 1.0 roughly doubles a member's degree with cover traffic.
        n_features: node feature dimension.
        feature_signal: mean shift applied to feature dim 0 for ring members.
            The only per-node signal there is; everything else is N(0, 1) noise.
            Kept small on purpose so a GNN has to use structure to beat an MLP.
        seed: RNG seed.

    Returns:
        FraudGraph with exact node and edge level ground truth.
    """
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}, expected one of {TOPOLOGIES}")
    if camouflage < 0:
        raise ValueError("camouflage must be >= 0")
    if topology == "bipartite" and ring_size < 2:
        raise ValueError("bipartite rings need ring_size >= 2")
    rng = np.random.default_rng(seed)

    bg_edges, pool = _barabasi_albert(n_background, m_attach, rng)
    edge_set = {(min(u, v), max(u, v)) for u, v in bg_edges}

    # Ring members are appended after the background -> contiguous high ids.
    n_ring_nodes = n_rings * ring_size
    n_total = n_background + n_ring_nodes
    ring_id = np.full(n_total, -1, dtype=np.int64)

    motif: list[tuple[int, int]] = []
    for r in range(n_rings):
        members = [n_background + r * ring_size + i for i in range(ring_size)]
        ring_id[members] = r
        motif.extend(_motif_edges(members, topology))
    motif = [(min(u, v), max(u, v)) for u, v in motif]

    collisions = sum(1 for e in motif if e in edge_set)  # always 0 by construction
    edge_set.update(motif)

    # Camouflage: preferential attachment from each ring member to legit nodes.
    ring_degree = np.zeros(n_total, dtype=np.int64)
    for u, v in motif:
        ring_degree[u] += 1
        ring_degree[v] += 1
    n_camo = 0
    for v in range(n_background, n_total):
        k = 1 + int(round(camouflage * ring_degree[v]))
        for u in _sample_distinct(pool, min(k, n_background), rng):
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                edge_set.add(e)
                n_camo += 1

    edges = np.array(sorted(edge_set), dtype=np.int64)
    y = (ring_id >= 0).astype(np.int64)

    x = rng.standard_normal((n_total, n_features)).astype(np.float32)
    x[y == 1, 0] += feature_signal

    return FraudGraph(
        edges=edges,
        x=x,
        y=y,
        ring_id=ring_id,
        ring_edges=np.array(sorted(set(motif)), dtype=np.int64).reshape(-1, 2),
        meta=dict(
            n_background=n_background,
            m_attach=m_attach,
            n_rings=n_rings,
            ring_size=ring_size,
            topology=topology,
            camouflage=camouflage,
            n_features=n_features,
            feature_signal=feature_signal,
            seed=seed,
            n_camouflage_edges=n_camo,
            n_motif_background_collisions=collisions,
        ),
    )
