# RingFaith

In GNN fraud detection the headline number is node-level AUC. An investigator
does not act on a score, though — they act on the *explained structure*, the
handful of edges the explainer says mattered. This repo measures how far apart
those two things drift.

Three quantities that usually get conflated:

1. **node detection** — AUC / average precision
2. **ring recovery** — did the model surface the planted collusion ring at all
3. **explanation faithfulness** — do the explainer's top edges recover the
   ring's *own* edges

(1) and (2) are prior art (see [positioning](#what-this-is-and-what-it-is-not)).
(3) is what I add, together with the random-edge null that any faithfulness
number has to be read against.

Everything below is measured. 80 configurations (4 ring topologies × 4
camouflage levels × 5 seeds), 240 trained models, 11,967 per-node faithfulness
measurements, 28 minutes on a laptop CPU. The numbers in every table are
regenerated from `reports/*.csv` by `experiments/make_tables.py`; none were
typed by hand.

## Findings

**F1 — Raw explanation overlap is mostly the null, not the explainer.**
Across the 16 topology × camouflage cells, GNNExplainer's raw top-k precision
correlates **r = +0.945** (p = 3.4e-08) with the *analytic random baseline* for
that cell. It is largely measuring how dense the candidate neighbourhood is.
Correcting for the null inverts the reading: on cliques, raw precision falls
0.660 → 0.144 as camouflage goes 0 → 2, while lift over the null *rises*
1.127 → 2.592 (peaking at camouflage 1.0). A faithfulness number reported
without its null can point the wrong way.

**F2 — Node AUC predicts raw faithfulness, but not faithfulness over the null.**
Across those same 16 cells, node AUC vs raw precision is **r = +0.769
(p = 0.0005)**; node AUC vs lift over the null is **r = −0.226 (p = 0.40, not
significant)**. The intuition that a better detector comes with a better
explanation survives only until you control for chance. This is the
dissociation the repo is about.

**F3 — Camouflage drives detection and faithfulness in opposite directions.**
In all four topologies node AUC falls monotonically with camouflage (Spearman
−1.0). Lift over the null rises (bipartite +1.0, star +0.8, clique +0.4, cycle
+0.2). Star is the cleanest case: AUC 0.915 → 0.596 while lift 1.130 → 2.048.
Harder detection, relatively *better*-than-chance explanations, and lower
absolute usefulness at the same time.

**F4 — Ring recovery collapses long before node AUC does.**
Clique at camouflage 2.0: **AUC 0.881, ring recall 0.100**. Bipartite at 2.0:
**AUC 0.729, ring recall 0.000**. Four of sixteen cells sit above 0.70 AUC with
ring recall below 0.20. A model can look acceptable on the headline metric
while returning nothing an investigator can open a case on.

**F5 — A plain gradient beats GNNExplainer here.**
Pooled lift over the null: **gradient 2.423, GNNExplainer 1.784, random 1.001**.
Paired on the same node and model, the gradient wins 42.6% of the time, ties
24.0%, and loses 33.4%, mean margin +0.046 (Wilcoxon p = 1.8e-27). The gradient
costs one backward pass; GNNExplainer costs 150 optimisation steps. I expected
the learned mask to win and it did not.

**F6 — The better detector carries the less faithful explainer.**
GCN reaches AUC 0.839 and ring recall 0.348; GraphSAGE reaches 0.698 and 0.142.
Yet gradient lift is **2.204 on GCN and 2.642 on GraphSAGE**. Picking a model on
detection alone picks against explanation quality in this setup.

**F7 — In aggregate the explainers beat chance; per node it is close to a coin flip.**
GNNExplainer on GCN beats its own null on **54.6%** of individual nodes;
gradient on GraphSAGE on **71.1%**. The aggregate margins are overwhelmingly
significant (p < 1e-60) because n is large, but a single explanation is not
something you would want to hand an investigator unchecked.

**F8 — Topology matters less than I expected (negative result).**
I set out expecting ring topology to be the main axis of dissociation. It is
not. Ranking the four topologies by node AUC and by lift over the null gives
almost the same order — only clique and bipartite swap the top two places.
Camouflage (F3), not topology, is where detection and faithfulness come apart.
Reporting this plainly rather than dressing it up.

**F9 — Instrument: the obvious split is fully degenerate.**
The generator appends ring members after the background graph, so their node ids
are contiguous at the top — exactly what happens when you build a graph in
construction order. A 60/20/20 contiguous split puts **0 of 48 fraud nodes in
train and all 48 in test**; AUC is undefined and the model cannot learn the
class at all. `stratified_split` on the same graph gives test AUC 0.988.
Reproduce with `make demo` (`reports/degenerate_split.json`).

**F10 — Instrument: the random control is calibrated.**
Over 3,989 measurements the random explainer scores mean precision **0.2293**
against an analytic expectation of **0.2297**, lift **1.001**. The null does what
it is supposed to do, which is what makes F1, F2 and F5 readable.

### What I did not find

I did not find a bug in this harness. I looked for the two I most expected.

The id-ordering trap (F9) is real, but I built the generator that way on purpose
and guarded it from the start with `assert_not_degenerate`, so it never
corrupted a result — it is a demonstration, not a postmortem.

The other was tie-breaking. Ring members hold the highest node ids and `edges`
is sorted by id, so any block of tied explainer scores ranked in index order
would push ring edges systematically to the back and bias faithfulness
downward. I wrote that up as a measured finding before measuring it. It is not
one: the observed tie rate for the gradient explainer in float32 is **0.000**,
and index-order versus random tie-breaking give identical precision to three
decimals on every topology. I deleted the claim and kept the random
tie-breaking as a guard, with a test pinning it. Recording the near-miss because
the wrong version of that paragraph was already written down.

## Results

`lift over null` is the column that matters: precision divided by the analytic random baseline for that cell. `random null` is that baseline. `nodes explained` is the support — faithfulness is measured only on fraud nodes the model actually scored above 0.5, since explaining a missed node asks a different question.

### Table 1 — GCN: detection vs explanation faithfulness

| topology   |   camouflage |   node AUC |   ring recall |   GNNExpl precision |   random null |   lift over null |   candidate edges |   nodes explained |
|:-----------|-------------:|-----------:|--------------:|--------------------:|--------------:|-----------------:|------------------:|------------------:|
| bipartite  |          0   |      0.996 |         0.9   |               0.61  |         0.545 |            1.238 |            34     |               125 |
| bipartite  |          0.5 |      0.95  |         0.567 |               0.366 |         0.253 |            1.621 |            73.072 |               125 |
| bipartite  |          1   |      0.833 |         0.167 |               0.273 |         0.148 |            2.071 |           126.416 |               125 |
| bipartite  |          2   |      0.729 |         0     |               0.197 |         0.086 |            2.345 |           243.568 |               125 |
| clique     |          0   |      0.999 |         1     |               0.66  |         0.611 |            1.127 |            49.304 |               125 |
| clique     |          0.5 |      0.97  |         0.667 |               0.352 |         0.187 |            1.994 |           165.608 |               125 |
| clique     |          1   |      0.903 |         0.4   |               0.325 |         0.128 |            2.592 |           274.712 |               125 |
| clique     |          2   |      0.881 |         0.1   |               0.144 |         0.104 |            1.434 |           626.848 |               125 |
| cycle      |          0   |      0.949 |         0.733 |               0.357 |         0.287 |            1.806 |            21.264 |               125 |
| cycle      |          0.5 |      0.871 |         0.333 |               0.186 |         0.148 |            1.485 |            37.376 |               125 |
| cycle      |          1   |      0.679 |         0.033 |               0.099 |         0.099 |            0.985 |            52.72  |               125 |
| cycle      |          2   |      0.65  |         0     |               0.101 |         0.054 |            2.18  |            94.816 |               125 |
| star       |          0   |      0.915 |         0.5   |               0.448 |         0.415 |            1.13  |            23.024 |               125 |
| star       |          0.5 |      0.82  |         0.167 |               0.299 |         0.294 |            1.037 |            34.512 |               125 |
| star       |          1   |      0.69  |         0     |               0.242 |         0.175 |            1.53  |            60.784 |               125 |
| star       |          2   |      0.596 |         0     |               0.174 |         0.109 |            2.048 |            87.112 |               125 |

### Table 2 — explainers, pooled over all cells

| model   | explainer    |   precision mean |   precision std |   random_expectation mean |   random_expectation std |   lift mean |   lift std |
|:--------|:-------------|-----------------:|----------------:|--------------------------:|-------------------------:|------------:|-----------:|
| gcn     | gnnexplainer |            0.302 |           0.22  |                     0.228 |                    0.184 |       1.664 |      1.712 |
| gcn     | grad         |            0.331 |           0.207 |                     0.228 |                    0.184 |       2.204 |      2.511 |
| gcn     | random       |            0.226 |           0.211 |                     0.228 |                    0.184 |       1.001 |      0.951 |
| sage    | gnnexplainer |            0.326 |           0.225 |                     0.232 |                    0.188 |       1.905 |      2.091 |
| sage    | grad         |            0.389 |           0.204 |                     0.232 |                    0.188 |       2.642 |      2.706 |
| sage    | random       |            0.233 |           0.217 |                     0.232 |                    0.188 |       1.001 |      0.909 |

### Table 3 — structure-blind baseline

| topology   |   auc (gcn) |   auc (mlp) |   auc (sage) |   ring_recall (gcn) |   ring_recall (mlp) |   ring_recall (sage) |
|:-----------|------------:|------------:|-------------:|--------------------:|--------------------:|---------------------:|
| bipartite  |       0.877 |       0.515 |        0.684 |               0.408 |                   0 |                0.15  |
| clique     |       0.938 |       0.555 |        0.803 |               0.542 |                   0 |                0.275 |
| cycle      |       0.787 |       0.542 |        0.619 |               0.275 |                   0 |                0.05  |
| star       |       0.755 |       0.518 |        0.687 |               0.167 |                   0 |                0.092 |

### Table 4 — the dissociation (averaged over camouflage)

| topology   |   node AUC |   GNNExpl precision |   lift over null |   AUC rank |   lift rank |
|:-----------|-----------:|--------------------:|-----------------:|-----------:|------------:|
| bipartite  |      0.877 |               0.362 |            1.819 |          2 |           1 |
| clique     |      0.938 |               0.37  |            1.787 |          1 |           2 |
| cycle      |      0.787 |               0.186 |            1.614 |          3 |           3 |
| star       |      0.755 |               0.291 |            1.436 |          4 |           4 |

### Table 5 — random-explainer control calibration

|   mean random precision |   mean analytic null |   mean lift (should be ~1.0) |   n measurements |
|------------------------:|---------------------:|-----------------------------:|-----------------:|
|                  0.2293 |               0.2297 |                        1.001 |             3989 |

### Table 6 — statistics quoted in the findings

| test                                                                   | stat               | p         |
|:-----------------------------------------------------------------------|:-------------------|:----------|
| pearson r, node AUC vs GNNExpl precision (n=16 cells)                  | 0.769              | 4.94e-04  |
| pearson r, node AUC vs lift over null (n=16 cells)                     | -0.226             | 4.00e-01  |
| pearson r, random null vs GNNExpl precision (n=16 cells)               | 0.945              | 3.35e-08  |
| wilcoxon, grad vs gnnexplainer precision (paired, n=3989)              | 0.0462             | 1.81e-27  |
| grad wins / ties / loses vs gnnexplainer (%)                           | 42.6 / 24.0 / 33.4 | -         |
| gcn gnnexplainer: beats own random null on % of nodes (n=2000)         | 54.6               | 2.15e-66  |
| gcn grad: beats own random null on % of nodes (n=2000)                 | 60.9               | 1.90e-101 |
| sage gnnexplainer: beats own random null on % of nodes (n=1989)        | 60.6               | 8.27e-84  |
| sage grad: beats own random null on % of nodes (n=1989)                | 71.1               | 4.51e-162 |
| bipartite: spearman(camouflage, node AUC) / spearman(camouflage, lift) | -1.0 / +1.0        | -         |
| clique: spearman(camouflage, node AUC) / spearman(camouflage, lift)    | -1.0 / +0.4        | -         |
| cycle: spearman(camouflage, node AUC) / spearman(camouflage, lift)     | -1.0 / +0.2        | -         |
| star: spearman(camouflage, node AUC) / spearman(camouflage, lift)      | -1.0 / +0.8        | -         |

## What this is, and what it is not

Ring-level ground truth for fraud graphs is not my idea. **TravelFraudBench**
(arXiv:2604.21093, Sajja, April 2026) already builds a configurable benchmark
with planted fraud rings in travel networks and already reports ring-level
recovery: GraphSAGE recovers 100% of rings across their ring types while a
tabular MLP recovers 17–88%. If you want a detection benchmark with ring
ground truth, use theirs. Motif-level supervision in money laundering is also
established — **LAS-GNN** (ACM ICAIF 2025, doi:10.1145/3768292.3770410) detects
temporal laundering motifs such as scatter-gather directly.

What those lines do not ask is whether the *explainer* attached to the detector
points at the ring. That is the only thing this repo adds: an edge-level
faithfulness layer measured against the planted motif's own edges, and against
a random-edge null. The generator here exists to make that measurable, not to
compete as a benchmark.

## Method

**Generator** (`src/ringfaith/generate.py`). A Barabási–Albert background graph
(Barabási & Albert, *Science* 286, 1999) of legitimate nodes, then `n_rings`
disjoint rings whose members are appended as new nodes. Four motif topologies:

| topology | motif edges per ring of size `s` | shape |
|---|---|---|
| `clique` | `s(s-1)/2` | everyone transacts with everyone |
| `star` | `s-1` | one hub, spokes |
| `cycle` | `s` | circular pass-through |
| `bipartite` | `⌊s/2⌋·⌈s/2⌉` | mule accounts ↔ merchants |

`camouflage` controls hiding: each ring member gets `1 + round(camouflage ×
d_ring)` extra edges to *legitimate* nodes, chosen by preferential attachment,
where `d_ring` is its degree inside the motif. At `camouflage=0` a member has
exactly one legitimate edge (enough to stay connected); at `2.0` its cover
traffic outnumbers its motif edges roughly two to one.

Node features are `N(0,1)` with a single mean shift of `feature_signal=0.35` on
dimension 0 for ring members. That shift is deliberately weak: it is the only
per-node signal, so a model that beats the structure-blind MLP has to be using
the graph. Ground truth is exact — node labels, ring ids, and the motif's own
edge list.

**Models** (`src/ringfaith/models.py`). GCN (arXiv:1609.02907), GraphSAGE with
mean aggregation (arXiv:1706.02216), and a structure-blind MLP baseline. All in
plain PyTorch on a dense adjacency; no torch-geometric. Two layers, hidden 32,
class-balanced cross-entropy, early stopping on validation loss.

**Explainers** (`src/ringfaith/explain.py`). All three score the *same*
candidate set — the edges of the target's 2-hop subgraph.

- `gnnexplainer` — a learned sigmoid edge mask (arXiv:1903.03894), optimised to
  preserve the target's predicted class under size and entropy penalties.
- `grad` — `|∂logit/∂edge_weight|` at unit weights. This is gradient×input on
  the edge weights, but since every input is exactly 1, the product degenerates
  to the plain absolute gradient. Worth stating rather than dressing up.
- `random` — uniform scores. The mandatory null.

The explainers do **not** extract a subgraph and run the model on it. They run
the model on the full graph and only let the mask vary over candidate edges,
leaving everything else at weight 1. That keeps the target's prediction exactly
equal to its full-graph prediction. Subgraph extraction does not: for a 2-layer
GCN, a 1-hop neighbour's symmetric normalisation depends on node degrees 3 hops
out, so a 2-hop extraction silently perturbs the very prediction being
explained.

**Metrics** (`src/ringfaith/metrics.py`). Node AUC and average precision; ring
recall (a ring counts as recovered when ≥80% of its members land in the top-K
nodes, K = the true fraud count — the ≥80% convention is TravelFraudBench's);
and edge faithfulness, the overlap between an explainer's top-k candidate edges
and the planted motif edges. The budget `k` defaults to the number of motif
edges in the candidate set, which makes precision = recall = F1, so one number
per node.

## Limitations

- **Synthetic only.** No real transaction graph. The generator is a controlled
  instrument for a specific question, not a claim about production data.
- **Undirected, static, unattributed edges.** Real fraud graphs are directed,
  timestamped and carry amounts. Motifs like scatter-gather are *defined* by
  direction and time; none of that is representable here.
- **One feature mechanism.** A single mean shift on one dimension. A different
  feature-signal design could move the detection numbers substantially.
- **Two explainers plus the null.** No PGExplainer, no SubgraphX, no attention.
  A negative result for GNNExplainer is not a negative result for all explainers.
- **Homophily by construction.** Ring members share a feature shift *and* are
  densely connected, which is the regime GNNs are best in. Heterophilous fraud
  (a mule that looks exactly like its legitimate neighbours) is not covered.
- **`k` is oracle-sized.** Defaulting the budget to the true number of motif
  edges in the neighbourhood is generous to the explainers; an investigator does
  not know that number in advance.
- **Small graphs.** Dense adjacency is `O(N²)`; everything here is under ~1.5k
  nodes. Nothing about scaling is tested.

## Reproduce

```bash
make venv     # .venv + editable install
make test     # 40 tests
make sweep    # the full topology x camouflage sweep
make demo     # the degenerate-split finding
make report   # rebuild the tables in this README from reports/*.csv
```

Every table above is regenerated by `experiments/make_tables.py` from the CSVs
in `reports/`, which are written by `experiments/run_sweep.py`. No number in
this README was typed by hand.

## Layout

```
src/ringfaith/
  generate.py   graph generator + exact ground truth
  models.py     GCN / GraphSAGE / MLP on a dense adjacency
  explain.py    GNNExplainer, gradient, random null
  metrics.py    AUC/AP, ring recall, edge faithfulness
  split.py      stratified split + degeneracy check
  experiment.py one experimental cell
experiments/    run_sweep.py, degenerate_split_demo.py, make_tables.py
tests/          40 pytest tests on the generator and the metrics
reports/        result CSVs and JSON written by the runs above
```

## References

Verified to exist at the time of writing; anything I could not resolve was left out.

- Sajja. *TravelFraudBench: A Configurable Evaluation Framework for GNN Fraud
  Ring Detection in Travel Networks.* arXiv:2604.21093, 2026.
- *LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif
  Detection.* ACM ICAIF 2025. doi:10.1145/3768292.3770410
- Ying, Bourgeois, You, Zitnik, Leskovec. *GNNExplainer: Generating
  Explanations for Graph Neural Networks.* arXiv:1903.03894, 2019.
- Kipf, Welling. *Semi-Supervised Classification with Graph Convolutional
  Networks.* arXiv:1609.02907, 2016.
- Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs.*
  arXiv:1706.02216, 2017.
- Barabási, Albert. *Emergence of Scaling in Random Networks.* Science 286, 1999.

## License

MIT.
