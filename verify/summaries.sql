-- Recompute reports/detection_summary.csv and reports/faithfulness_summary.csv
-- from the per-run and per-node raw tables, in SQLite.
--
-- Both summaries are written by experiments/run_sweep.py with one pandas
-- groupby each. Nothing checked that groupby: the tests exercise the metric
-- functions, and every consumer downstream reads the summary rather than the
-- raw rows. This derives all 48 detection rows and all 1512 faithfulness rows
-- again with nothing but SQL, so an error in the aggregation would have to be
-- reproduced here to survive.
--
-- The published values are rounded to 4 decimals, so a recomputed value may
-- legitimately differ by up to 5e-5, half a step of the last published digit.
-- The threshold here is 6e-5; anything past that is a real disagreement.
--
-- Run through verify/verify.sh, which decompresses the faithfulness table and
-- flattens the two-level pandas header on the summaries.

.bail on
.mode csv
.headers off

.import --csv reports/detection_raw.csv det
.import --csv verify/.work/faithfulness_raw.csv fai
.import --csv verify/.work/det_summary_flat.csv det_pub
.import --csv verify/.work/fai_summary_flat.csv fai_pub

-- Sample standard deviation, ddof=1, which is what pandas .std() returns.
-- Written from the sums because SQLite has no aggregate for it.
CREATE TEMP VIEW det_calc AS
SELECT topology,
       CAST(camouflage AS REAL) AS camo,
       model,
       AVG(CAST(auc AS REAL))                                       AS auc_mean,
       SQRT((SUM(CAST(auc AS REAL) * CAST(auc AS REAL))
             - COUNT(*) * AVG(CAST(auc AS REAL)) * AVG(CAST(auc AS REAL)))
            / (COUNT(*) - 1.0))                                     AS auc_std,
       AVG(CAST(ap AS REAL))                                        AS ap_mean,
       SQRT((SUM(CAST(ap AS REAL) * CAST(ap AS REAL))
             - COUNT(*) * AVG(CAST(ap AS REAL)) * AVG(CAST(ap AS REAL)))
            / (COUNT(*) - 1.0))                                     AS ap_std,
       AVG(CAST(ring_recall AS REAL))                               AS rr_mean,
       SQRT((SUM(CAST(ring_recall AS REAL) * CAST(ring_recall AS REAL))
             - COUNT(*) * AVG(CAST(ring_recall AS REAL)) * AVG(CAST(ring_recall AS REAL)))
            / (COUNT(*) - 1.0))                                     AS rr_std,
       COUNT(*)                                                     AS n
FROM det
GROUP BY topology, camo, model;

CREATE TEMP VIEW fai_calc AS
SELECT topology,
       CAST(camouflage AS REAL) AS camo,
       model, explainer,
       CAST(detected AS INT) AS detected,
       k_mode,
       AVG(CAST(precision AS REAL))                                 AS p_mean,
       SQRT((SUM(CAST(precision AS REAL) * CAST(precision AS REAL))
             - COUNT(*) * AVG(CAST(precision AS REAL)) * AVG(CAST(precision AS REAL)))
            / (COUNT(*) - 1.0))                                     AS p_std,
       AVG(CAST(random_expectation AS REAL))                        AS r_mean,
       SQRT((SUM(CAST(random_expectation AS REAL) * CAST(random_expectation AS REAL))
             - COUNT(*) * AVG(CAST(random_expectation AS REAL)) * AVG(CAST(random_expectation AS REAL)))
            / (COUNT(*) - 1.0))                                     AS r_std,
       AVG(CAST(lift AS REAL))                                      AS l_mean,
       SQRT((SUM(CAST(lift AS REAL) * CAST(lift AS REAL))
             - COUNT(*) * AVG(CAST(lift AS REAL)) * AVG(CAST(lift AS REAL)))
            / (COUNT(*) - 1.0))                                     AS l_std,
       AVG(CAST(n_candidates AS REAL))                              AS c_mean,
       SQRT((SUM(CAST(n_candidates AS REAL) * CAST(n_candidates AS REAL))
             - COUNT(*) * AVG(CAST(n_candidates AS REAL)) * AVG(CAST(n_candidates AS REAL)))
            / (COUNT(*) - 1.0))                                     AS c_std,
       AVG(CAST(n_relevant AS REAL))                                AS v_mean,
       SQRT((SUM(CAST(n_relevant AS REAL) * CAST(n_relevant AS REAL))
             - COUNT(*) * AVG(CAST(n_relevant AS REAL)) * AVG(CAST(n_relevant AS REAL)))
            / (COUNT(*) - 1.0))                                     AS v_std,
       COUNT(*)                                                     AS n
FROM fai
GROUP BY topology, camo, model, explainer, detected, k_mode;

-- One row per published row, carrying the largest disagreement on it.
CREATE TEMP VIEW det_cmp AS
SELECT c.topology || ' ' || c.camo || ' ' || c.model AS key,
       MAX(ABS(c.auc_mean - CAST(p.auc_mean AS REAL)),
           ABS(c.auc_std  - CAST(p.auc_std  AS REAL)),
           ABS(c.ap_mean  - CAST(p.ap_mean  AS REAL)),
           ABS(c.ap_std   - CAST(p.ap_std   AS REAL)),
           ABS(c.rr_mean  - CAST(p.ring_recall_mean AS REAL)),
           ABS(c.rr_std   - CAST(p.ring_recall_std  AS REAL))) AS dev
FROM det_calc c
JOIN det_pub p
  ON p.topology = c.topology
 AND CAST(p.camouflage AS REAL) = c.camo
 AND p.model = c.model;

CREATE TEMP VIEW fai_cmp AS
SELECT c.topology || ' ' || c.camo || ' ' || c.model || ' ' || c.explainer
       || ' ' || c.detected || ' ' || c.k_mode AS key,
       MAX(ABS(c.p_mean - CAST(p.precision_mean AS REAL)),
           ABS(c.p_std  - CAST(p.precision_std  AS REAL)),
           ABS(c.r_mean - CAST(p.random_expectation_mean AS REAL)),
           ABS(c.r_std  - CAST(p.random_expectation_std  AS REAL)),
           ABS(c.l_mean - CAST(p.lift_mean AS REAL)),
           ABS(c.l_std  - CAST(p.lift_std  AS REAL)),
           ABS(c.c_mean - CAST(p.n_candidates_mean AS REAL)),
           ABS(c.c_std  - CAST(p.n_candidates_std  AS REAL)),
           ABS(c.v_mean - CAST(p.n_relevant_mean AS REAL)),
           ABS(c.v_std  - CAST(p.n_relevant_std  AS REAL))) AS dev
FROM fai_calc c
JOIN fai_pub p
  ON p.topology = c.topology
 AND CAST(p.camouflage AS REAL) = c.camo
 AND p.model = c.model
 AND p.explainer = c.explainer
 AND CAST(p.detected AS INT) = c.detected
 AND p.k_mode = c.k_mode;

.mode list
.separator |

-- The worst rows, so the measured agreement is visible on a pass as well.
SELECT 'worst', key, printf('%.6e', dev) FROM det_cmp ORDER BY dev DESC LIMIT 3;
SELECT 'worst', key, printf('%.6e', dev) FROM fai_cmp ORDER BY dev DESC LIMIT 3;

-- name, rows recomputed, rows published, rows joined, max deviation, rows past tolerance
SELECT 'detection_summary',
       (SELECT COUNT(*) FROM det_calc),
       (SELECT COUNT(*) FROM det_pub),
       (SELECT COUNT(*) FROM det_cmp),
       printf('%.6e', (SELECT MAX(dev) FROM det_cmp)),
       (SELECT COUNT(*) FROM det_cmp WHERE dev > 6e-5);
SELECT 'faithfulness_summary',
       (SELECT COUNT(*) FROM fai_calc),
       (SELECT COUNT(*) FROM fai_pub),
       (SELECT COUNT(*) FROM fai_cmp),
       printf('%.6e', (SELECT MAX(dev) FROM fai_cmp)),
       (SELECT COUNT(*) FROM fai_cmp WHERE dev > 6e-5);
