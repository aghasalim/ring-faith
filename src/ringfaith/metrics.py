"""Detection, ring-recovery and explanation-faithfulness metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def node_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """Node level AUC and average precision -- the usual headline numbers."""
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {"auc": float(roc_auc_score(y, scores)), "ap": float(average_precision_score(y, scores))}


def ring_recall(y: np.ndarray, ring_id: np.ndarray, scores: np.ndarray, frac: float = 0.8) -> float:
    """Fraction of planted rings with >= `frac` of members inside the top-K nodes.

    K is the true number of fraud nodes. The >=80% definition follows
    TravelFraudBench (arXiv:2604.21093); ring-level recovery is their idea, not
    a contribution of this repo.
    """
    k = int(y.sum())
    if k == 0:
        return float("nan")
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-scores, kind="stable")[:k]] = True
    rings = [r for r in np.unique(ring_id) if r >= 0]
    if not rings:
        return float("nan")
    return float(np.mean([flagged[ring_id == r].mean() >= frac for r in rings]))


def _rank_desc(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """argsort descending with *random* tie-breaking.

    Defensive, not corrective. `edges` is sorted by node id and ring members
    hold the highest ids, so any tied block ranked in index order would put ring
    edges systematically last and bias faithfulness downward. Measured tie rate
    for the gradient explainer on these graphs is 0.000 in float32, so this
    changes no reported number here; it is kept because the bias would be
    silent if a future explainer (or a coarser dtype) did produce ties.
    `tests/test_metrics.py::test_all_tied_scores_are_broken_without_index_order_bias`
    pins the unbiased behaviour.
    """
    perm = rng.permutation(len(scores))
    return perm[np.argsort(-scores[perm], kind="stable")]


def edge_faithfulness(
    cand: np.ndarray,
    scores: np.ndarray,
    ring_edge_mask: np.ndarray,
    k: int | None = None,
    seed: int = 0,
) -> dict | None:
    """Overlap between an explainer's top-k candidate edges and the planted motif.

    Args:
        cand: indices into the global edge array that the explainer scored.
        scores: explainer score per candidate, same order and length as `cand`.
        ring_edge_mask: [E] bool, True where a global edge is a planted motif edge.
        k: explanation budget. Defaults to the number of relevant edges in the
            candidate set, which makes precision == recall == F1.
        seed: seed for tie-breaking.

    Returns None if the candidate set contains no motif edge (the metric is
    undefined there), otherwise precision/recall/f1 plus the analytic random
    expectation and the lift over it.
    """
    relevant = ring_edge_mask[cand]
    n_rel = int(relevant.sum())
    if n_rel == 0 or len(cand) == 0:
        return None
    k = n_rel if k is None else min(int(k), len(cand))
    order = _rank_desc(np.asarray(scores, dtype=float), np.random.default_rng(seed))
    hits = int(relevant[order[:k]].sum())
    precision, recall = hits / k, hits / n_rel
    expected = n_rel / len(cand)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "random_expectation": expected,
        "lift": precision / expected if expected > 0 else float("nan"),
        "n_candidates": int(len(cand)),
        "n_relevant": n_rel,
        "k": int(k),
    }
