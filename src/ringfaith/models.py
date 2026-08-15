"""GNNs on a dense adjacency, in plain PyTorch. No torch-geometric dependency.

Every model takes `(x, edges, edge_weight)` where `edges` is [E, 2] undirected
and `edge_weight` is [E] (default all ones). The adjacency is built from
`edge_weight` differentiably, which is what lets the explainers in
`ringfaith.explain` optimise or differentiate a mask over edges. The mask lives
on *undirected* edges, so it is symmetric by construction -- the same
granularity the edge-level ground truth is defined at.

Dense is O(N^2) memory. Fine for the graphs here (<2k nodes); it would not be
for a real transaction graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_adj(edges: torch.Tensor, n: int, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric dense adjacency [n, n] from undirected `edges` [E, 2].

    Differentiable with respect to `edge_weight`.
    """
    u, v = edges[:, 0], edges[:, 1]
    if edge_weight is None:
        edge_weight = torch.ones(len(edges), dtype=torch.float32, device=edges.device)
    a = torch.zeros(n, n, dtype=edge_weight.dtype, device=edge_weight.device)
    a = a.index_put((u, v), edge_weight, accumulate=True)
    a = a.index_put((v, u), edge_weight, accumulate=True)
    return a


def gcn_norm(a: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation of A + I (Kipf & Welling)."""
    a = a + torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    d = a.sum(1).clamp(min=1e-10).pow(-0.5)
    return d.unsqueeze(1) * a * d.unsqueeze(0)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edges, edge_weight=None):
        a = gcn_norm(dense_adj(edges, x.shape[0], edge_weight))
        h = F.relu(self.w1(a @ x))
        h = F.dropout(h, self.dropout, self.training)
        return self.w2(a @ h)


class SAGE(nn.Module):
    """GraphSAGE with mean aggregation, separate self and neighbour weights."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.self1, self.neigh1 = nn.Linear(in_dim, hidden), nn.Linear(in_dim, hidden, bias=False)
        self.self2, self.neigh2 = nn.Linear(hidden, out_dim), nn.Linear(hidden, out_dim, bias=False)
        self.dropout = dropout

    @staticmethod
    def _mean(a, h):
        return (a @ h) / a.sum(1, keepdim=True).clamp(min=1e-10)

    def forward(self, x, edges, edge_weight=None):
        a = dense_adj(edges, x.shape[0], edge_weight)
        h = F.relu(self.self1(x) + self.neigh1(self._mean(a, x)))
        h = F.dropout(h, self.dropout, self.training)
        return self.self2(h) + self.neigh2(self._mean(a, h))


class MLP(nn.Module):
    """Structure-blind baseline. Ignores `edges` entirely."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )

    def forward(self, x, edges=None, edge_weight=None):
        return self.net(x)


MODELS = {"gcn": GCN, "sage": SAGE, "mlp": MLP}


def train(
    model: nn.Module,
    x: torch.Tensor,
    edges: torch.Tensor,
    y: torch.Tensor,
    train_idx,
    val_idx,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
):
    """Full-batch training with class-balanced loss and early stopping on val loss.

    Rings are a small minority, so the loss is weighted by inverse class
    frequency; without it the models collapse to predicting all-legitimate.
    """
    counts = torch.bincount(y[train_idx], minlength=2).float().clamp(min=1.0)
    weight = counts.sum() / (2.0 * counts)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edges)[train_idx], y[train_idx], weight=weight)
        loss.backward()
        opt.step()
        model.train(False)
        with torch.no_grad():
            vloss = F.cross_entropy(model(x, edges)[val_idx], y[val_idx], weight=weight).item()
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)
    return model


@torch.no_grad()
def fraud_scores(model: nn.Module, x, edges) -> torch.Tensor:
    """P(fraud) per node."""
    model.train(False)
    return F.softmax(model(x, edges), dim=1)[:, 1]
