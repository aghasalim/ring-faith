"""Edge explainers, all scoring the same candidate edge set.

Design note: the explainers do **not** extract a subgraph and run the model on
it. They run the model on the *full* graph and only let the mask vary over the
target's k-hop candidate edges; every other edge stays at weight 1. That keeps
the target's prediction exactly equal to its prediction on the full graph, which
subgraph extraction does not guarantee (GCN's degree normalisation makes a
1-hop neighbour's normalised row depend on edges 3 hops out). It costs a full
forward pass per step, which is cheap at this graph size.

All three explainers score the identical candidate set, so the random control is
an honest null for the other two.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPLAINERS = ("gnnexplainer", "grad", "random")


def adjacency_list(edges: np.ndarray, n: int) -> list[np.ndarray]:
    """Neighbour ids per node. Build once, reuse across targets."""
    order = np.argsort(edges[:, 0], kind="stable")
    both = np.concatenate([edges, edges[:, ::-1]])
    both = both[np.argsort(both[:, 0], kind="stable")]
    starts = np.searchsorted(both[:, 0], np.arange(n + 1))
    del order
    return [both[starts[i] : starts[i + 1], 1] for i in range(n)]


def khop_nodes(adj: list[np.ndarray], target: int, hops: int) -> np.ndarray:
    seen = {int(target)}
    frontier = {int(target)}
    for _ in range(hops):
        nxt: set[int] = set()
        for u in frontier:
            nxt.update(int(w) for w in adj[u])
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return np.fromiter(seen, dtype=np.int64, count=len(seen))


def candidate_edges(edges: np.ndarray, adj: list[np.ndarray], n: int, target: int, hops: int = 2) -> np.ndarray:
    """Indices into `edges` of the target's k-hop subgraph edges.

    Both endpoints must be within `hops` of the target. This is the shared
    candidate pool for every explainer and the denominator of the random null.
    """
    inside = np.zeros(n, dtype=bool)
    inside[khop_nodes(adj, target, hops)] = True
    return np.flatnonzero(inside[edges[:, 0]] & inside[edges[:, 1]])


def _predicted_class(model, x, edges, target: int) -> int:
    with torch.no_grad():
        return int(model(x, edges)[target].argmax())


def gnnexplainer(
    model,
    x: torch.Tensor,
    edges: torch.Tensor,
    target: int,
    cand: np.ndarray,
    epochs: int = 150,
    lr: float = 0.05,
    coeff_size: float = 0.005,
    coeff_entropy: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Ying et al. 2019 (arXiv:1903.03894): learn a sigmoid edge mask that keeps
    the target's predicted class while being small and near-binary."""
    model.train(False)
    cls = _predicted_class(model, x, edges, target)
    cand_t = torch.as_tensor(cand, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    param = nn.Parameter(torch.randn(len(cand), generator=gen) * 0.1)
    base = torch.ones(edges.shape[0])
    opt = torch.optim.Adam([param], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        m = torch.sigmoid(param)
        w = base.index_put((cand_t,), m)
        loss = -F.log_softmax(model(x, edges, w)[target], dim=-1)[cls]
        ent = -m * torch.log(m + 1e-12) - (1 - m) * torch.log(1 - m + 1e-12)
        (loss + coeff_size * m.sum() + coeff_entropy * ent.mean()).backward()
        opt.step()
    return torch.sigmoid(param).detach().numpy()


def grad(model, x: torch.Tensor, edges: torch.Tensor, target: int, cand: np.ndarray, **_) -> np.ndarray:
    """|d logit / d edge_weight| at unit weights.

    This is gradient-times-input on the edge weights; since the inputs are all
    exactly 1, the product degenerates to the plain absolute gradient.
    """
    model.train(False)
    w = torch.ones(edges.shape[0], requires_grad=True)
    logits = model(x, edges, w)
    cls = int(logits[target].argmax())
    g = torch.autograd.grad(logits[target, cls], w)[0]
    return g.abs().numpy()[cand]


def random(model, x, edges, target: int, cand: np.ndarray, seed: int = 0, **_) -> np.ndarray:
    """Mandatory null. Uniform scores over the same candidate set.

    Its expected precision@k is analytically `n_relevant / n_candidates` when
    k = n_relevant; `tests/test_metrics.py` checks the empirical value matches.
    """
    return np.random.default_rng(seed + 9973 * int(target)).random(len(cand))


REGISTRY = {"gnnexplainer": gnnexplainer, "grad": grad, "random": random}
