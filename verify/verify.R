# Independent statistical check of the inference in Table 6, in base R.
#
# Table 6 is the only place the repository turns its measurements into claims
# about significance: three Pearson correlations over the sixteen GCN cells,
# a paired Wilcoxon on four thousand nodes, six more against the random-explainer
# null, and a Spearman per topology. All of them come out of one call each to
# scipy in experiments/make_tables.py. This redoes them in base R and separates
# what must match exactly from what cannot:
#
#   deterministic  the correlation coefficients, the mean margin, the win, tie
#                  and loss shares and every n, which must agree to the decimal
#                  the table is published at
#   convention     the Wilcoxon p-values, which are normal approximations whose
#                  tie correction depends on which paired differences count as
#                  equal, and that is decided at the last bit of a float, so
#                  only the order of magnitude is comparable
#
# It then adds inference the repository does not have. Every published mean is
# pooled over nodes that share a graph and a trained model, so the spread quoted
# next to it is not an uncertainty on the mean. The cluster bootstrap at the end
# resamples whole configurations, which is the honest unit, and asks whether the
# headline explainer gap survives it.
#
# No packages, so CI needs nothing beyond the R that is already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

DRAWS <- 2000
TOL3 <- 6e-4        # columns published to 3 decimals
TOL4 <- 6e-5        # the mean margin, 4 decimals
TOL_PCT <- 0.06     # win / tie / loss shares, 1 decimal
TOL_R_REL <- 5e-3   # correlation p-values, published to 3 significant figures
TOL_P_LOG <- 0.02   # Wilcoxon p, compared as a relative difference in log10

failures <- 0
worst <- c(three_decimals = 0, four_decimals = 0, one_decimal = 0, p_value_log10 = 0)

fail <- function(msg) {
    cat("  FAIL:", msg, "\n")
    failures <<- failures + 1
}

check <- function(label, got, want, tol, bucket = "three_decimals") {
    delta <- abs(got - want)
    if (delta > worst[[bucket]]) worst[[bucket]] <<- delta
    ok <- delta <= tol
    if (!ok) failures <<- failures + 1
    cat(sprintf("  %-62s %12.5f  published %-10s %s\n", label, got, format(want),
                if (ok) "ok" else "FAIL"))
    invisible(ok)
}

# The paired signed-rank test, written out rather than called, so this is a
# second implementation of the statistic and not a second call to a library.
# Zero differences are dropped and ties are corrected, which is what scipy does;
# no continuity correction, for the same reason.
signed_rank <- function(x, y) {
    d <- x - y
    d <- d[d != 0]
    n <- length(d)
    r <- rank(abs(d))
    v <- sum(r[d > 0])
    counts <- table(r)
    sigma <- sqrt(n * (n + 1) * (2 * n + 1) / 24 - sum(counts^3 - counts) / 48)
    z <- (v - n * (n + 1) / 4) / sigma
    list(p = 2 * pnorm(-abs(z)), n = n, distinct = length(unique(abs(d))))
}

# One table out of reports/tables.md, found by the heading above it.
md_table <- function(path, heading) {
    lines <- readLines(path, warn = FALSE)
    start <- which(startsWith(lines, heading))
    if (length(start) == 0) stop("no heading ", heading, " in ", path)
    out <- list()
    for (line in lines[(start[1] + 1):length(lines)]) {
        s <- trimws(line)
        if (!startsWith(s, "|")) {
            if (length(out) > 0) break
            next
        }
        cells <- trimws(strsplit(gsub("^\\||\\|$", "", s), "|", fixed = TRUE)[[1]])
        if (gsub("[-:]", "", paste(cells, collapse = "")) == "") next
        out[[length(out) + 1]] <- cells
    }
    if (length(out) < 2) stop("no table under ", heading)
    df <- as.data.frame(do.call(rbind, out[-1]), stringsAsFactors = FALSE)
    names(df) <- out[[1]]
    df
}

# The row of Table 6 whose test column starts with `prefix`.
t6_row <- function(t6, prefix) {
    hit <- which(startsWith(t6$test, prefix))
    if (length(hit) != 1) stop("Table 6 has ", length(hit), " rows starting ", prefix)
    t6[hit[1], ]
}

raw_path <- file.path(root, "verify", ".work", "faithfulness_raw.csv")
if (!file.exists(raw_path)) {
    cat("cannot open", raw_path, "\n")
    cat("run verify/verify.sh, which unpacks the gzipped table\n")
    quit(status = 2)
}
det <- read.csv(file.path(root, "reports", "detection_raw.csv"))
fai <- read.csv(raw_path)
t6 <- md_table(file.path(root, "reports", "tables.md"), "### Table 6.")

oracle <- fai[fai$k_mode == "oracle" & fai$detected == 1, ]

# ------------------------------------------------------------------ the 16 cells
# make_tables.py rounds Table 1 to three decimals and then correlates those
# rounded values, so the same rounding is applied before the correlation here.
gcn_det <- det[det$model == "gcn", ]
gcn_fai <- oracle[oracle$model == "gcn" & oracle$explainer == "gnnexplainer", ]
cell <- paste(gcn_det$topology, gcn_det$camouflage)
cell_f <- paste(gcn_fai$topology, gcn_fai$camouflage)
keys <- sort(unique(cell))
h <- data.frame(
    topology = sub(" .*", "", keys),
    camouflage = as.numeric(sub(".* ", "", keys)),
    auc = round(tapply(gcn_det$auc, cell, mean)[keys], 3),
    precision = round(tapply(gcn_fai$precision, cell_f, mean)[keys], 3),
    null = round(tapply(gcn_fai$random_expectation, cell_f, mean)[keys], 3),
    lift = round(tapply(gcn_fai$lift, cell_f, mean)[keys], 3),
    row.names = NULL
)
if (nrow(h) != 16) stop("expected 16 GCN cells, found ", nrow(h))

cat("pearson correlations over the", nrow(h), "GCN cells\n")
pearson <- list(
    c("pearson r, node AUC vs GNNExpl precision", "auc", "precision"),
    c("pearson r, node AUC vs lift over null", "auc", "lift"),
    c("pearson r, random null vs GNNExpl precision", "null", "precision")
)
for (spec in pearson) {
    row <- t6_row(t6, spec[1])
    ct <- cor.test(h[[spec[2]]], h[[spec[3]]], method = "pearson")
    check(spec[1], unname(ct$estimate), as.numeric(row$stat), TOL3)
    want_p <- as.numeric(row$p)
    rel <- abs(ct$p.value - want_p) / want_p
    ok <- rel <= TOL_R_REL
    if (!ok) failures <- failures + 1
    cat(sprintf("  %-62s %12.3e  published %-10s %s\n",
                paste(spec[1], "p"), ct$p.value, row$p, if (ok) "ok" else "FAIL"))
}

# --------------------------------------------------------- paired over explainers
# One row per node that every explainer scored, which is what pivot_table
# followed by dropna leaves behind on the Python side.
node_key <- paste(oracle$topology, oracle$camouflage, oracle$seed,
                  oracle$model, oracle$node)
wide <- reshape(
    data.frame(node = node_key, model = oracle$model,
               explainer = oracle$explainer, precision = oracle$precision),
    idvar = c("node", "model"), timevar = "explainer", direction = "wide"
)
names(wide) <- sub("^precision\\.", "", names(wide))
explainers <- c("gnnexplainer", "grad", "ig", "random")
wide <- wide[complete.cases(wide[, explainers]), ]

cat("\ngrad against gnnexplainer, paired on", nrow(wide), "nodes\n")
row <- t6_row(t6, "wilcoxon, grad vs gnnexplainer")
if (!grepl(paste0("n=", nrow(wide)), row$test, fixed = TRUE)) {
    fail(sprintf("Table 6 says %s but %d nodes are paired here", row$test, nrow(wide)))
}
d <- wide$grad - wide$gnnexplainer
check("mean precision margin, grad - gnnexplainer", mean(d),
      as.numeric(row$stat), TOL4, "four_decimals")

compare_p <- function(label, got, want_text) {
    want <- as.numeric(want_text)
    rel <- abs(log10(got) - log10(want)) / abs(log10(want))
    if (rel > worst[["p_value_log10"]]) worst[["p_value_log10"]] <<- rel
    ok <- rel <= TOL_P_LOG
    if (!ok) failures <<- failures + 1
    cat(sprintf("  %-62s %12.3e  published %-10s %s\n", label, got, want_text,
                if (ok) "ok" else "FAIL"))
}
sr <- signed_rank(wide$grad, wide$gnnexplainer)
compare_p("wilcoxon p, grad vs gnnexplainer", sr$p, row$p)
cat(sprintf("  %-62s %12d\n", "non-zero paired differences", sr$n))
cat(sprintf("  %-62s %12d\n", "distinct absolute differences among them", sr$distinct))

row <- t6_row(t6, "grad wins / ties / loses")
shares <- as.numeric(trimws(strsplit(row$stat, "/", fixed = TRUE)[[1]]))
check("grad wins over gnnexplainer (%)", 100 * mean(d > 0), shares[1], TOL_PCT, "one_decimal")
check("grad ties with gnnexplainer (%)", 100 * mean(d == 0), shares[2], TOL_PCT, "one_decimal")
check("grad loses to gnnexplainer (%)", 100 * mean(d < 0), shares[3], TOL_PCT, "one_decimal")

cat("\neach explainer against its own random null\n")
for (model in sort(unique(wide$model))) {
    q <- wide[wide$model == model, ]
    for (ex in c("gnnexplainer", "grad", "ig")) {
        row <- t6_row(t6, sprintf("%s %s: beats own random null", model, ex))
        if (!grepl(paste0("n=", nrow(q)), row$test, fixed = TRUE)) {
            fail(sprintf("Table 6 says %s but %d nodes are paired here",
                         row$test, nrow(q)))
        }
        check(sprintf("%s %s beats its own null (%% of nodes)", model, ex),
              100 * mean(q[[ex]] > q$random), as.numeric(row$stat), TOL_PCT, "one_decimal")
        compare_p(sprintf("%s %s vs its own null, wilcoxon p", model, ex),
                  signed_rank(q[[ex]], q$random)$p, row$p)
    }
}

cat("\nspearman against camouflage, per topology\n")
for (topo in sort(unique(h$topology))) {
    row <- t6_row(t6, sprintf("%s: spearman", topo))
    want <- as.numeric(trimws(strsplit(row$stat, "/", fixed = TRUE)[[1]]))
    s <- h[h$topology == topo, ]
    s <- s[order(s$camouflage), ]
    check(sprintf("%s, spearman(camouflage, node AUC)", topo),
          cor(s$camouflage, s$auc, method = "spearman"), want[1], TOL_PCT, "one_decimal")
    check(sprintf("%s, spearman(camouflage, lift over null)", topo),
          cor(s$camouflage, s$lift, method = "spearman"), want[2], TOL_PCT, "one_decimal")
}

# ------------------------------------------------------- inference that is missing
# Every published mean pools nodes that share a graph and a trained model, so the
# standard deviation printed beside it is a spread over nodes, not an uncertainty
# on the mean. Resampling whole configurations is the honest unit: 80 of them,
# 4 topologies x 4 camouflage levels x 5 seeds.
cat("\ncluster bootstrap over configurations,", DRAWS, "draws\n")
config <- paste(oracle$topology, oracle$camouflage, oracle$seed)
configs <- unique(config)
by_config <- split(seq_len(nrow(oracle)), config)

boot_ci <- function(mask, column) {
    values <- oracle[[column]]
    stats <- numeric(DRAWS)
    for (b in seq_len(DRAWS)) {
        idx <- unlist(by_config[sample(configs, length(configs), replace = TRUE)],
                      use.names = FALSE)
        idx <- idx[mask[idx]]
        stats[b] <- mean(values[idx])
    }
    quantile(stats, c(0.025, 0.975), names = FALSE)
}

t2 <- md_table(file.path(root, "reports", "tables.md"), "### Table 2.")
for (spec in list(c("gcn", "ig"), c("gcn", "gnnexplainer"))) {
    model <- spec[1]
    ex <- spec[2]
    mask <- oracle$model == model & oracle$explainer == ex
    pub <- t2[t2$model == model & t2$explainer == ex, ]
    if (nrow(pub) != 1) {
        fail(sprintf("no Table 2 row for %s %s", model, ex))
        next
    }
    for (column in c("precision", "lift")) {
        ci <- boot_ci(mask, column)
        point <- as.numeric(pub[[paste(column, "mean")]])
        inside <- point >= ci[1] && point <= ci[2]
        if (!inside) failures <- failures + 1
        cat(sprintf("  %-30s published %.3f  95%% CI [%.3f, %.3f]  width %.3f  %s\n",
                    paste(model, ex, column), point, ci[1], ci[2], ci[2] - ci[1],
                    if (inside) "ok" else "FAIL"))
    }
}

# The gap the repository is built on: does integrated gradients still beat
# GNNExplainer once whole configurations are resampled rather than nodes?
gap <- numeric(DRAWS)
ig_mask <- oracle$model == "gcn" & oracle$explainer == "ig"
gn_mask <- oracle$model == "gcn" & oracle$explainer == "gnnexplainer"
for (b in seq_len(DRAWS)) {
    idx <- unlist(by_config[sample(configs, length(configs), replace = TRUE)],
                  use.names = FALSE)
    gap[b] <- mean(oracle$precision[idx[ig_mask[idx]]]) -
              mean(oracle$precision[idx[gn_mask[idx]]])
}
ci <- quantile(gap, c(0.025, 0.975), names = FALSE)
cat(sprintf("  %-30s mean %+.4f  95%% CI [%+.4f, %+.4f]\n",
            "gcn ig - gnnexplainer precision", mean(gap), ci[1], ci[2]))
if (ci[1] <= 0) {
    fail("the integrated-gradients advantage does not survive resampling configurations")
} else {
    cat("  the advantage survives resampling whole configurations\n")
}

cat(sprintf("\nworst |recomputed - published|, columns published to 3 decimals: %.3e\n",
            worst[["three_decimals"]]))
cat(sprintf("worst |recomputed - published|, columns published to 4 decimals: %.3e\n",
            worst[["four_decimals"]]))
cat(sprintf("worst |recomputed - published|, columns published to 1 decimal:  %.3e\n",
            worst[["one_decimal"]]))
cat(sprintf("worst relative difference in log10(p): %.4f\n", worst[["p_value_log10"]]))
if (failures > 0) {
    cat(sprintf("%d checks failed\n", failures))
    quit(status = 1)
}
cat("R reproduces Table 6, and the explainer gap survives a cluster bootstrap\n")
