#!/usr/bin/env bash
# Recompute the published tables in every language here and require agreement.
#
# Every number in the README comes out of pandas, in experiments/make_tables.py,
# from CSVs written by experiments/run_sweep.py. If one of those aggregations
# were wrong, nothing downstream would notice, because everything downstream
# reads the same output. The tests check that the code runs, not that it is
# right.
#
# These are independent recomputations from the raw per-run and per-node tables.
# An arithmetic mistake would have to be made identically in eight languages to
# survive. Each is skipped with a clear message if its toolchain is absent, so
# this runs on a laptop with only some of them installed. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

work="verify/.work"
mkdir -p "$work"

# The per-node table is committed gzipped, and sqlite, C and Rust have no way to
# read that. It is unpacked fresh on every run so a stale copy cannot hide a
# change to the committed one.
if ! gzip -dc reports/faithfulness_raw.csv.gz > "$work/faithfulness_raw.csv"; then
    echo "cannot unpack reports/faithfulness_raw.csv.gz"
    exit 1
fi

# The two summary CSVs carry a pandas two-level column header, which .import
# cannot parse. Only the header is rewritten; the numbers are untouched, and
# verify/gocheck checks that the names below still describe the real file.
{
    printf 'topology,camouflage,model,auc_mean,auc_std,ap_mean,ap_std,'
    printf 'ring_recall_mean,ring_recall_std\n'
    tail -n +4 reports/detection_summary.csv
} > "$work/det_summary_flat.csv"
{
    printf 'topology,camouflage,model,explainer,detected,k_mode,'
    printf 'precision_mean,precision_std,random_expectation_mean,'
    printf 'random_expectation_std,lift_mean,lift_std,n_candidates_mean,'
    printf 'n_candidates_std,n_relevant_mean,n_relevant_std\n'
    tail -n +4 reports/faithfulness_summary.csv
} > "$work/fai_summary_flat.csv"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL has no assertion of its own, so its output is checked here. The last two
# lines carry the row counts, the largest deviation and how many rows sit past
# the tolerance.
check_sql () {
    local out
    out=$(sqlite3 -init verify/summaries.sql :memory: "" 2>&1) || { echo "$out"; return 1; }
    echo "$out"
    echo "$out" | awk -F'|' '
        /^(detection|faithfulness)_summary\|/ {
            seen++
            if ($2 != $3 || $2 != $4) {
                printf "%s: recomputed %s rows, published %s, joined %s\n", $1, $2, $3, $4
                bad++
            }
            if ($6 + 0 > 0) {
                printf "%s: %s rows disagree past the tolerance\n", $1, $6
                bad++
            }
        }
        END {
            if (seen != 2) { print "the SQL did not report both summaries"; bad++ }
            exit bad > 0 ? 1 : 0
        }'
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$work/kernel" verify/kernel.c -lm || return 1
    "$work/kernel" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/nullcheck && cargo run --release --quiet -- "$root" ); }

run "SQL, the two summary tables"      sqlite3 check_sql
run "C, the per-row metric kernel"     cc      check_c
run "Go, file and README validation"   go      check_go
run "R, the inference in table 6"      Rscript Rscript verify/verify.R "$root"
run "Rust, the analytic null"          cargo   check_rust
run "JavaScript, the budget sweep"     node    node verify/budget.mjs "$root"
run "Ruby, the detected/missed split"  ruby    ruby verify/tables37.rb "$root"
run "Java, the pooled comparison"      java    java verify/Pooled.java "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
