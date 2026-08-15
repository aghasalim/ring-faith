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
