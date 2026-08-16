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
| gcn     | ig           |            0.366 |           0.2   |                     0.228 |                    0.184 |       2.566 |      2.803 |
| gcn     | random       |            0.226 |           0.211 |                     0.228 |                    0.184 |       1.001 |      0.951 |
| sage    | gnnexplainer |            0.326 |           0.225 |                     0.232 |                    0.188 |       1.905 |      2.091 |
| sage    | grad         |            0.389 |           0.204 |                     0.232 |                    0.188 |       2.642 |      2.706 |
| sage    | ig           |            0.389 |           0.204 |                     0.232 |                    0.188 |       2.642 |      2.706 |
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
| gcn ig: beats own random null on % of nodes (n=2000)                   | 69.0               | 6.68e-147 |
| sage gnnexplainer: beats own random null on % of nodes (n=1989)        | 60.6               | 8.27e-84  |
| sage grad: beats own random null on % of nodes (n=1989)                | 71.1               | 4.51e-162 |
| sage ig: beats own random null on % of nodes (n=1989)                  | 71.1               | 4.51e-162 |
| bipartite: spearman(camouflage, node AUC) / spearman(camouflage, lift) | -1.0 / +1.0        | -         |
| clique: spearman(camouflage, node AUC) / spearman(camouflage, lift)    | -1.0 / +0.4        | -         |
| cycle: spearman(camouflage, node AUC) / spearman(camouflage, lift)     | -1.0 / +0.2        | -         |
| star: spearman(camouflage, node AUC) / spearman(camouflage, lift)      | -1.0 / +0.8        | -         |

### Table 7 — faithfulness on detected vs missed fraud nodes (oracle budget)

| model   | explainer    |   detected lift |   detected precision |   detected n |   missed lift |   missed precision |   missed n |   lift difference |   paired cells |   wilcoxon p |
|:--------|:-------------|----------------:|---------------------:|-------------:|--------------:|-------------------:|-----------:|------------------:|---------------:|-------------:|
| gcn     | gnnexplainer |           1.664 |                0.302 |         2000 |         1.158 |              0.194 |        466 |             0.505 |             67 |     3.18e-05 |
| gcn     | grad         |           2.204 |                0.331 |         2000 |         2.611 |              0.279 |        466 |            -0.407 |             67 |     0.129    |
| gcn     | ig           |           2.566 |                0.366 |         2000 |         3.328 |              0.327 |        466 |            -0.762 |             67 |     0.365    |
| gcn     | random       |           1.001 |                0.226 |         2000 |         0.992 |              0.17  |        466 |             0.009 |             67 |     0.508    |
| sage    | gnnexplainer |           1.905 |                0.326 |         1989 |         1.655 |              0.25  |       1009 |             0.25  |             79 |     2.03e-05 |
| sage    | grad         |           2.642 |                0.389 |         1989 |         3.071 |              0.362 |       1009 |            -0.429 |             79 |     0.324    |
| sage    | ig           |           2.642 |                0.389 |         1989 |         3.071 |              0.362 |       1009 |            -0.429 |             79 |     0.324    |
| sage    | random       |           1.001 |                0.233 |         1989 |         0.989 |              0.185 |       1009 |             0.012 |             79 |     0.503    |

### Table 8 — mean lift over the null by explanation budget (detected nodes)

| explainer    |    k1 |    k3 |    k5 |   k10 |   k20 |   oracle |
|:-------------|------:|------:|------:|------:|------:|---------:|
| gnnexplainer | 1.92  | 2.058 | 2.182 | 2.156 | 1.831 |    1.784 |
| grad         | 3.002 | 2.896 | 2.718 | 2.313 | 1.865 |    2.423 |
| ig           | 3.242 | 3.258 | 3.075 | 2.548 | 1.975 |    2.604 |
| random       | 0.922 | 0.976 | 0.979 | 1.005 | 1.008 |    1.001 |

### Table 9 — mean lift over the null by explanation budget (missed nodes)

| explainer    |    k1 |    k3 |    k5 |   k10 |   k20 |   oracle |
|:-------------|------:|------:|------:|------:|------:|---------:|
| gnnexplainer | 1.109 | 1.307 | 1.459 | 1.768 | 1.77  |    1.498 |
| grad         | 3.527 | 3.361 | 3.263 | 2.77  | 2.169 |    2.926 |
| ig           | 4.182 | 3.854 | 3.595 | 2.986 | 2.253 |    3.152 |
| random       | 0.895 | 0.951 | 0.992 | 1.025 | 1.022 |    0.99  |

### Table 10 — does the gradient still beat GNNExplainer at a realistic budget?

| budget   | comparison           |   challenger lift |   gnnexpl lift |   mean precision margin | wins / ties / losses (%)   |   wilcoxon p |    n |
|:---------|:---------------------|------------------:|---------------:|------------------------:|:---------------------------|-------------:|-----:|
| k1       | grad vs gnnexplainer |             3.002 |          1.92  |                  0.0719 | 26.4 / 54.3 / 19.3         |     1.79e-11 | 3989 |
| k3       | grad vs gnnexplainer |             2.896 |          2.058 |                  0.0592 | 40.5 / 31.7 / 27.9         |     1.77e-09 | 3989 |
| k5       | grad vs gnnexplainer |             2.718 |          2.182 |                  0.029  | 40.6 / 27.2 / 32.2         |     2.44e-10 | 3989 |
| k10      | grad vs gnnexplainer |             2.313 |          2.156 |                 -0.0019 | 34.5 / 29.3 / 36.2         |     0.718    | 3989 |
| k20      | grad vs gnnexplainer |             1.865 |          1.831 |                 -0.0044 | 31.4 / 37.8 / 30.8         |     0.0207   | 3989 |
| oracle   | grad vs gnnexplainer |             2.423 |          1.784 |                  0.0462 | 42.6 / 24.0 / 33.4         |     1.81e-27 | 3989 |
| k1       | ig vs gnnexplainer   |             3.242 |          1.92  |                  0.1176 | 28.2 / 55.3 / 16.5         |     1.16e-28 | 3989 |
| k3       | ig vs gnnexplainer   |             3.258 |          2.058 |                  0.1113 | 44.6 / 32.8 / 22.5         |     4.46e-47 | 3989 |
| k5       | ig vs gnnexplainer   |             3.075 |          2.182 |                  0.0772 | 46.2 / 27.7 / 26.1         |     5.43e-66 | 3989 |
| k10      | ig vs gnnexplainer   |             2.548 |          2.156 |                  0.0277 | 39.9 / 29.9 / 30.2         |     3.78e-25 | 3989 |
| k20      | ig vs gnnexplainer   |             1.975 |          1.831 |                  0.0086 | 35.0 / 39.2 / 25.7         |     2.23e-10 | 3989 |
| oracle   | ig vs gnnexplainer   |             2.604 |          1.784 |                  0.0637 | 47.1 / 24.4 / 28.6         |     7.46e-60 | 3989 |
