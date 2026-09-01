/* Recompute the faithfulness metric kernel for every measured node, in C.
 *
 * src/ringfaith/metrics.py::edge_faithfulness writes five numbers per row:
 * precision, recall, f1, the analytic random-edge expectation and the lift
 * over it. Four of those five are functions of the three counts it also
 * writes, so they can be derived again from the counts alone:
 *
 *     random_expectation = n_relevant / n_candidates
 *     recall             = precision * k / n_relevant
 *     f1                 = 2 * precision * recall / (precision + recall)
 *     lift               = precision / random_expectation
 *
 * The whole argument in the README is built on lift, and lift was published by
 * exactly one implementation. This is the second one, over all 131136 rows.
 *
 * It also recomputes the Table 5 null-calibration row, which is the check the
 * repository offers on the analytic null, and compares it against the numbers
 * actually published in reports/tables.md.
 *
 * Columns are resolved by name, so a column added upstream cannot silently
 * shift what this reads.
 */
/* strtok_r is POSIX, and -std=c99 hides it on glibc without this. */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 4096
#define TOL 1e-9         /* the identities are exact bar float rounding */
#define PUB_TOL 6e-5     /* table 5 is published to 4 decimals */

static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    strncpy(buf, header, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';

    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[256];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p)
            return "";
        p++;
        col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out)
        n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl)
        *nl = '\0';
    return out;
}

/* Pull one numbered single-row table out of reports/tables.md. The published
 * numbers live in that file and nowhere else, so they are read rather than
 * copied into this source, where they would rot. */
static int read_table5(const char *path, double *out, int want)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[LINE];
    int state = 0, got = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "### Table 5.", 12) == 0) { state = 1; continue; }
        if (state == 0)
            continue;
        if (line[0] != '|')
            continue;
        state++;                     /* 2 = header row, 3 = separator, 4 = data */
        if (state < 4)
            continue;
        char *save = NULL;
        for (char *tok = strtok_r(line, "|", &save); tok; tok = strtok_r(NULL, "|", &save)) {
            while (*tok == ' ') tok++;
            if (*tok == '\0' || *tok == '\n')
                continue;
            if (got < want)
                out[got++] = atof(tok);
        }
        break;
    }
    fclose(f);
    return got == want ? 0 : -1;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE];

    snprintf(path, sizeof path, "%s/verify/.work/faithfulness_raw.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        fprintf(stderr, "run verify/verify.sh, which unpacks the gzipped table\n");
        return 2;
    }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }

    const char *names[] = {"explainer", "detected", "k_mode", "precision", "recall",
                           "f1", "random_expectation", "lift", "n_candidates",
                           "n_relevant", "k"};
    int col[11];
    for (int i = 0; i < 11; i++) {
        col[i] = column_of(header, names[i]);
        if (col[i] < 0) {
            fprintf(stderr, "no column %s in faithfulness_raw.csv\n", names[i]);
            fclose(f);
            return 2;
        }
    }

    double worst[4] = {0, 0, 0, 0};
    const char *what[4] = {"random_expectation", "recall", "f1", "lift"};
    long rows = 0, bad = 0;

    /* Table 5: the random explainer at the oracle budget on detected nodes. */
    double sum_p = 0, sum_r = 0, sum_l = 0;
    long n_cal = 0;

    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0')
            continue;
        char explainer[64], k_mode[32];
        strncpy(explainer, field(line, col[0]), sizeof explainer - 1);
        explainer[sizeof explainer - 1] = '\0';
        const int detected = atoi(field(line, col[1]));
        strncpy(k_mode, field(line, col[2]), sizeof k_mode - 1);
        k_mode[sizeof k_mode - 1] = '\0';

        const double precision = atof(field(line, col[3]));
        const double recall = atof(field(line, col[4]));
        const double f1 = atof(field(line, col[5]));
        const double rand_exp = atof(field(line, col[6]));
        const double lift = atof(field(line, col[7]));
        const double n_cand = atof(field(line, col[8]));
        const double n_rel = atof(field(line, col[9]));
        const double k = atof(field(line, col[10]));

        if (n_cand <= 0 || n_rel <= 0 || k <= 0) {
            fprintf(stderr, "row %ld has a non-positive count\n", rows + 2);
            bad++;
            rows++;
            continue;
        }

        const double my_exp = n_rel / n_cand;
        const double my_recall = precision * k / n_rel;
        const double denom = precision + my_recall;
        const double my_f1 = denom == 0.0 ? 0.0 : 2.0 * precision * my_recall / denom;
        const double my_lift = precision / my_exp;

        const double d[4] = {fabs(my_exp - rand_exp), fabs(my_recall - recall),
                             fabs(my_f1 - f1), fabs(my_lift - lift)};
        for (int i = 0; i < 4; i++) {
            if (d[i] > worst[i])
                worst[i] = d[i];
            if (d[i] > TOL)
                bad++;
        }

        if (strcmp(explainer, "random") == 0 && detected == 1
            && strcmp(k_mode, "oracle") == 0) {
            sum_p += precision;
            sum_r += rand_exp;
            sum_l += lift;
            n_cal++;
        }
        rows++;
    }
    fclose(f);

    printf("checked %ld rows of faithfulness_raw.csv\n", rows);
    for (int i = 0; i < 4; i++)
        printf("  %-20s max |recomputed - published| %.3e  %s\n",
               what[i], worst[i], worst[i] > TOL ? "FAIL" : "ok");

    if (n_cal == 0) {
        fprintf(stderr, "no random-explainer oracle rows on detected nodes\n");
        return 1;
    }

    snprintf(path, sizeof path, "%s/reports/tables.md", root);
    double pub[4];
    if (read_table5(path, pub, 4) != 0) {
        fprintf(stderr, "cannot read the Table 5 row from %s\n", path);
        return 2;
    }

    const double got[4] = {sum_p / n_cal, sum_r / n_cal, sum_l / n_cal, (double)n_cal};
    const char *label[4] = {"mean random precision", "mean analytic null",
                            "mean lift", "n measurements"};
    printf("\ntable 5, the random-explainer control, from %ld measurements\n", n_cal);
    for (int i = 0; i < 4; i++) {
        const double tol = i == 3 ? 0.0 : PUB_TOL;
        const double delta = fabs(got[i] - pub[i]);
        const int fail = delta > tol;
        bad += fail;
        printf("  %-22s recomputed %.6f  published %.4f  |d| %.1e  %s\n",
               label[i], got[i], pub[i], delta, fail ? "FAIL" : "ok");
    }

    if (bad) {
        printf("\n%ld disagreements\n", bad);
        return 1;
    }
    printf("\nC reproduces every per-row metric and the Table 5 calibration row\n");
    return 0;
}
