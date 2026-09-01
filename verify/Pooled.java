// Recompute Table 2 and Table 10 from the per-node raw table, in Java.
//
// Table 2 is the pooled explainer comparison, the eight rows the abstract
// quotes when it says integrated gradients reaches 41% precision against a 23%
// null. Table 10 is the head-to-head that decides whether the gradient still
// beats GNNExplainer once the budget stops being an oracle, which is the test
// that can overturn the repository's own earlier finding. Both were produced by
// one pandas aggregation each, and nothing else recomputed them.
//
// The Wilcoxon p in Table 10 is not checked here; that column is checked in
// verify/verify.R. Everything that is a mean, a margin, a share or a count is
// checked here.
//
// Usage: java verify/Pooled.java <repo root>

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Pooled {

    static final double TOL3 = 6e-4;    // columns published to 3 decimals
    static final double TOL4 = 6e-5;    // the precision margin, 4 decimals
    static final double TOL_PCT = 0.06; // win/tie/loss shares, 1 decimal
    static final String[] BUDGETS = {"k1", "k3", "k5", "k10", "k20", "oracle"};

    static int failures = 0;
    // Worst deviation per published precision, so a column printed to one
    // decimal does not hide behind one printed to four.
    static final double[] worst = new double[3];
    static final String[] BUCKET = {"3 decimals", "4 decimals", "1 decimal"};

    static boolean check(double got, double want, double tol, int bucket) {
        double delta = Math.abs(got - want);
        if (delta > worst[bucket]) worst[bucket] = delta;
        boolean bad = delta > tol;
        if (bad) failures++;
        return bad;
    }

    /** Running mean and sample standard deviation, ddof=1, as pandas reports it. */
    static final class Stat {
        final List<Double> values = new ArrayList<>();

        void add(double v) { values.add(v); }

        double mean() {
            double s = 0;
            for (double v : values) s += v;
            return s / values.size();
        }

        double std() {
            double m = mean(), s = 0;
            for (double v : values) s += (v - m) * (v - m);
            return Math.sqrt(s / (values.size() - 1));
        }
    }

    static final class Table {
        List<String> header;
        List<List<String>> rows = new ArrayList<>();
    }

    /** One table out of reports/tables.md, found by the heading above it. */
    static Table mdTable(List<String> lines, String heading) {
        int start = -1;
        for (int i = 0; i < lines.size(); i++) {
            if (lines.get(i).startsWith(heading)) { start = i; break; }
        }
        if (start < 0) throw new IllegalStateException("no heading " + heading);

        List<List<String>> out = new ArrayList<>();
        for (int i = start + 1; i < lines.size(); i++) {
            String s = lines.get(i).trim();
            if (!s.startsWith("|")) {
                if (!out.isEmpty()) break;
                continue;
            }
            String body = s.replaceAll("^\\|", "").replaceAll("\\|$", "");
            List<String> cells = new ArrayList<>();
            for (String c : body.split("\\|", -1)) cells.add(c.trim());
            if (String.join("", cells).replaceAll("[-:]", "").isEmpty()) continue;
            out.add(cells);
        }
        if (out.size() < 2) throw new IllegalStateException("no table under " + heading);
        Table t = new Table();
        t.header = out.get(0);
        t.rows = out.subList(1, out.size());
        return t;
    }

    public static void main(String[] args) throws IOException {
        String root = args.length > 0 ? args[0] : ".";
        Path raw = Path.of(root, "verify", ".work", "faithfulness_raw.csv");
        if (!Files.exists(raw)) {
            System.err.println("cannot open " + raw);
            System.err.println("run verify/verify.sh, which unpacks the gzipped table");
            System.exit(2);
        }
        List<String> tables = Files.readAllLines(Path.of(root, "reports", "tables.md"));

        // Pooled statistics for table 2, and the paired precisions for table 10,
        // in one pass over the 131k rows.
        Map<String, Stat[]> pooled = new LinkedHashMap<>();
        Map<String, Stat> liftByBudget = new HashMap<>();
        Map<String, Map<String, double[]>> paired = new HashMap<>();
        List<String> explainers = new ArrayList<>();

        try (BufferedReader r = Files.newBufferedReader(raw)) {
            String[] header = r.readLine().split(",", -1);
            Map<String, Integer> at = new HashMap<>();
            for (int i = 0; i < header.length; i++) at.put(header[i], i);
            for (String name : new String[]{"topology", "camouflage", "seed", "model",
                    "explainer", "node", "detected", "k_mode", "precision",
                    "random_expectation", "lift"}) {
                if (!at.containsKey(name)) {
                    System.err.println("no column " + name + " in faithfulness_raw.csv");
                    System.exit(2);
                }
            }
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] f = line.split(",", -1);
                if (!f[at.get("detected")].equals("1")) continue;

                String explainer = f[at.get("explainer")];
                String kMode = f[at.get("k_mode")];
                double precision = Double.parseDouble(f[at.get("precision")]);
                double lift = Double.parseDouble(f[at.get("lift")]);

                if (!explainers.contains(explainer)) explainers.add(explainer);
                liftByBudget.computeIfAbsent(kMode + "|" + explainer, k -> new Stat()).add(lift);

                String node = String.join("|", f[at.get("topology")], f[at.get("camouflage")],
                        f[at.get("seed")], f[at.get("model")], f[at.get("node")]);
                Map<String, double[]> perNode =
                        paired.computeIfAbsent(kMode, k -> new LinkedHashMap<>());
                double[] slot = perNode.get(node);
                if (slot == null) {
                    slot = new double[8];
                    Arrays.fill(slot, Double.NaN);
                    perNode.put(node, slot);
                }
                int idx = explainers.indexOf(explainer);
                slot[idx] = precision;

                if (kMode.equals("oracle")) {
                    Stat[] s = pooled.computeIfAbsent(
                            f[at.get("model")] + "|" + explainer,
                            k -> new Stat[]{new Stat(), new Stat(), new Stat()});
                    s[0].add(precision);
                    s[1].add(Double.parseDouble(f[at.get("random_expectation")]));
                    s[2].add(lift);
                }
            }
        }

        // ------------------------------------------------------------ table 2
        Table pub2 = mdTable(tables, "### Table 2.");
        System.out.println("table 2, explainers pooled over all cells");
        for (List<String> row : pub2.rows) {
            Stat[] s = pooled.get(row.get(0) + "|" + row.get(1));
            if (s == null) {
                System.out.printf("  %s/%s: no rows in the raw data%n", row.get(0), row.get(1));
                failures++;
                continue;
            }
            boolean bad = false;
            double[] got = {s[0].mean(), s[0].std(), s[1].mean(), s[1].std(),
                            s[2].mean(), s[2].std()};
            for (int i = 0; i < 6; i++) {
                bad |= check(got[i], Double.parseDouble(row.get(i + 2)), TOL3, 0);
            }
            System.out.printf("  %-5s %-13s precision %.4f (%.4f)  null %.4f (%.4f)  "
                            + "lift %.4f (%.4f)  n %d  %s%n",
                    row.get(0), row.get(1), got[0], got[1], got[2], got[3], got[4], got[5],
                    s[0].values.size(), bad ? "FAIL" : "ok");
        }

        // ----------------------------------------------------------- table 10
        int gnn = explainers.indexOf("gnnexplainer");
        Table pub10 = mdTable(tables, "### Table 10.");
        System.out.println("\ntable 10, the challengers against GNNExplainer at every budget");
        for (List<String> row : pub10.rows) {
            String budget = row.get(0);
            String challenger = row.get(1).split(" ")[0];
            int ci = explainers.indexOf(challenger);
            Map<String, double[]> perNode = paired.get(budget);
            if (perNode == null || ci < 0 || gnn < 0) {
                System.out.printf("  %s %s: no rows in the raw data%n", budget, row.get(1));
                failures++;
                continue;
            }
            int n = 0, wins = 0, ties = 0, losses = 0;
            double marginSum = 0;
            for (double[] v : perNode.values()) {
                boolean complete = true;
                for (int i = 0; i < explainers.size(); i++) {
                    if (Double.isNaN(v[i])) complete = false;
                }
                if (!complete) continue;
                double d = v[ci] - v[gnn];
                marginSum += d;
                if (d > 0) wins++;
                else if (d == 0) ties++;
                else losses++;
                n++;
            }
            if (n == 0) {
                System.out.printf("  %s %s: nothing paired%n", budget, row.get(1));
                failures++;
                continue;
            }
            double margin = marginSum / n;
            double challengerLift = liftByBudget.get(budget + "|" + challenger).mean();
            double gnnLift = liftByBudget.get(budget + "|gnnexplainer").mean();
            String[] shares = row.get(5).split("/");

            boolean bad = false;
            bad |= check(challengerLift, Double.parseDouble(row.get(2)), TOL3, 0);
            bad |= check(gnnLift, Double.parseDouble(row.get(3)), TOL3, 0);
            bad |= check(margin, Double.parseDouble(row.get(4)), TOL4, 1);
            bad |= check(100.0 * wins / n, Double.parseDouble(shares[0].trim()), TOL_PCT, 2);
            bad |= check(100.0 * ties / n, Double.parseDouble(shares[1].trim()), TOL_PCT, 2);
            bad |= check(100.0 * losses / n, Double.parseDouble(shares[2].trim()), TOL_PCT, 2);
            bad |= check(n, Double.parseDouble(row.get(7)), 0, 0);

            System.out.printf("  %-6s %-22s lift %.4f vs %.4f  margin %+.5f  "
                            + "%.1f/%.1f/%.1f  n %d  %s%n",
                    budget, row.get(1), challengerLift, gnnLift, margin,
                    100.0 * wins / n, 100.0 * ties / n, 100.0 * losses / n, n,
                    bad ? "FAIL" : "ok");
        }

        System.out.println();
        for (int i = 0; i < worst.length; i++) {
            System.out.printf("worst |recomputed - published|, columns published to %-10s %.3e%n",
                    BUCKET[i], worst[i]);
        }
        if (failures > 0) {
            System.out.printf("%d disagreements%n", failures);
            System.exit(1);
        }
        System.out.println("Java reproduces the pooled comparison and the budget head-to-head");
    }
}
