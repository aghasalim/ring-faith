// Structural validation of every results file, plus a second implementation of
// the headline table.
//
// The CSVs under reports/ are the evidence for every number in the README, and
// nothing checked that they are well formed. A truncated write, a column that
// drifted, a NaN that leaked out of a division: all of it would be invisible
// until someone read the table. This walks every one of them, checks the row
// counts against what reports/sweep_config.json claims was run, recomputes
// Table 1 from the raw rows, and finally checks that the tables pasted into
// README.md are still the tables reports/tables.md generates.
package main

import (
	"compress/gzip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Table 1 is published to 3 decimals, so half a step is 5e-4. The threshold is
// a shade above it so a value that rounded half-to-even is not a failure.
const tableTol = 6e-4

type table struct {
	header []string
	rows   [][]string
	index  map[string]int
}

func (t *table) col(name string) int {
	if i, ok := t.index[name]; ok {
		return i
	}
	return -1
}

func (t *table) num(row []string, name string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(row[t.col(name)]), 64)
	if err != nil {
		return math.NaN()
	}
	return v
}

func readCSV(r io.Reader) (*table, error) {
	c := csv.NewReader(r)
	c.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := c.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("only %d rows", len(rows))
	}
	t := &table{header: rows[0], rows: rows[1:], index: map[string]int{}}
	for i, h := range rows[0] {
		t.index[h] = i
	}
	return t, nil
}

func open(path string) (*table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if strings.HasSuffix(path, ".gz") {
		z, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer z.Close()
		return readCSV(z)
	}
	return readCSV(f)
}

// pandas writes a grouped frame with a two level column header and the index
// names on a third line. Those files get their shape and their cells checked
// like any other, but the column-name rules are applied to the composed names.
func multiIndex(t *table) bool { return t.header[0] == "" }

func composed(t *table) []string {
	if len(t.rows) < 2 {
		return nil
	}
	level1, names := t.rows[0], t.rows[1]
	out := make([]string, 0, len(t.header))
	for i := range t.header {
		if names[i] != "" {
			out = append(out, names[i])
		} else {
			out = append(out, t.header[i]+"_"+level1[i])
		}
	}
	return out
}

func validate(path string, t *table) []string {
	var problems []string

	names := t.header
	skipHeader := multiIndex(t)
	if skipHeader {
		names = composed(t)
	}
	seen := map[string]bool{}
	for _, h := range names {
		if h == "" {
			problems = append(problems, "a column has an empty name")
		} else if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	start := 0
	if skipHeader {
		start = 2 // the two rows that finish the header
	}
	for i := start; i < len(t.rows); i++ {
		for j, cell := range t.rows[i] {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is empty", i+2, names[j]))
				continue
			}
			if low == "nan" || low == "inf" || low == "-inf" || low == "+inf" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, names[j], cell))
			}
		}
	}
	return problems
}

type key struct {
	topology   string
	camouflage float64
}

type acc struct {
	sum []float64
	n   int
}

func (a *acc) add(v []float64) {
	if a.sum == nil {
		a.sum = make([]float64, len(v))
	}
	for i := range v {
		a.sum[i] += v[i]
	}
	a.n++
}

func (a *acc) mean(i int) float64 { return a.sum[i] / float64(a.n) }

// mdTables pulls every pipe table out of a markdown file as trimmed cells.
// Separator rows are dropped: the dashes and alignment colons are formatting,
// and the two files format them differently.
func mdTables(path string) ([][][]string, error) {
	text, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][][]string
	var cur [][]string
	for _, line := range strings.Split(string(text), "\n") {
		s := strings.TrimSpace(line)
		if !strings.HasPrefix(s, "|") {
			if len(cur) > 0 {
				out = append(out, cur)
				cur = nil
			}
			continue
		}
		cells := strings.Split(strings.Trim(s, "|"), "|")
		for i := range cells {
			cells[i] = strings.TrimSpace(cells[i])
		}
		if strings.Trim(strings.Join(cells, ""), "-:") == "" {
			continue
		}
		cur = append(cur, cells)
	}
	if len(cur) > 0 {
		out = append(out, cur)
	}
	return out, nil
}

func sameTable(a, b [][]string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

// A recomputed number is compared against the prose by printing it to the same
// number of decimals the prose used. That is exactly the claim being made, and
// it needs no tolerance: "0.71" says two decimals, "78%" says none.
func like(got float64, text string) string {
	decimals := 0
	if dot := strings.IndexByte(text, '.'); dot >= 0 {
		decimals = len(text) - dot - 1
	}
	return strconv.FormatFloat(got, 'f', decimals, 64)
}

type meanAcc struct {
	sum float64
	n   int
}

func (m *meanAcc) add(v float64) { m.sum += v; m.n++ }
func (m *meanAcc) mean() float64 { return m.sum / float64(m.n) }

func meansBy(rows [][]string, t *table, keep func([]string) bool,
	key func([]string) string, value string) map[string]float64 {
	acc := map[string]*meanAcc{}
	for _, r := range rows {
		if !keep(r) {
			continue
		}
		k := key(r)
		if acc[k] == nil {
			acc[k] = &meanAcc{}
		}
		acc[k].add(t.num(r, value))
	}
	out := map[string]float64{}
	for k, a := range acc {
		out[k] = a.mean()
	}
	return out
}

func checkProse(root string, det, fai *table) int {
	raw, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		fmt.Printf("  README.md: %v\n", err)
		return 1
	}
	// The claims run across line breaks, so the text is compared as one line.
	prose := strings.Join(strings.Fields(string(raw)), " ")

	isGCN := func(r []string) bool { return r[det.col("model")] == "gcn" }
	camo := func(r []string) string { return r[det.col("camouflage")] }
	gcnAUC := meansBy(det.rows, det, isGCN, camo, "auc")
	gcnRecall := meansBy(det.rows, det, isGCN, camo, "ring_recall")
	gcnRingByCell := meansBy(det.rows, det, isGCN,
		func(r []string) string { return r[det.col("topology")] + " " + camo(r) }, "ring_recall")
	mlpAUC := meansBy(det.rows, det, func(r []string) bool { return r[det.col("model")] == "mlp" },
		func(r []string) string { return r[det.col("topology")] }, "auc")

	detected := func(r []string) bool { return r[fai.col("detected")] == "1" }
	explainerIs := func(name string) func([]string) bool {
		return func(r []string) bool { return detected(r) && r[fai.col("explainer")] == name }
	}
	gcnExplainer := func(name string) func([]string) bool {
		return func(r []string) bool {
			return detected(r) && r[fai.col("model")] == "gcn" &&
				r[fai.col("explainer")] == name
		}
	}
	one := func(r []string) string { return "all" }
	faiCamo := func(r []string) string { return r[fai.col("camouflage")] }

	igPrecision := meansBy(fai.rows, fai, explainerIs("ig"), one, "precision")["all"]
	igNull := meansBy(fai.rows, fai, explainerIs("ig"), one, "random_expectation")["all"]
	gnnLift := meansBy(fai.rows, fai, gcnExplainer("gnnexplainer"), faiCamo, "lift")
	igLift := meansBy(fai.rows, fai, gcnExplainer("ig"), faiCamo, "lift")

	// Cell means rounded the way Table 1 rounds them, since the sentence is a
	// claim about the published cells.
	round3 := func(x float64) float64 { return math.Round(x*1000) / 1000 }
	cellAUC := meansBy(det.rows, det, func(r []string) bool { return true },
		func(r []string) string {
			return r[det.col("topology")] + " " + camo(r) + " " + r[det.col("model")]
		}, "auc")
	cellRecall := meansBy(det.rows, det, func(r []string) bool { return true },
		func(r []string) string {
			return r[det.col("topology")] + " " + camo(r) + " " + r[det.col("model")]
		}, "ring_recall")

	type claim struct {
		pattern string
		got     []float64
	}
	claims := []claim{
		{`still scores ([0-9.]+) AUC, which reads as a working model, while ring ` +
			`recovery has fallen from ([0-9.]+)% to ([0-9.]+)%`,
			[]float64{gcnAUC["2.0"], 100 * gcnRecall["0.0"], 100 * gcnRecall["2.0"]}},
		{`puts ([0-9.]+)% of its top edges inside the ring against a ([0-9.]+)% ` +
			`random-edge expectation`,
			[]float64{100 * igPrecision, 100 * igNull}},
		{`GNNExplainer goes from ([0-9.]+)x at camouflage 0 to ([0-9.]+)x at ` +
			`camouflage 2.0. Integrated Gradients goes from ([0-9.]+)x to ([0-9.]+)x, ` +
			`with its peak of ([0-9.]+)x at camouflage 1.0`,
			[]float64{gnnLift["0.0"], gnnLift["2.0"], igLift["0.0"], igLift["2.0"],
				igLift["1.0"]}},
		{`node AUC only slides from ([0-9.]+) to ([0-9.]+)`,
			[]float64{gcnAUC["0.0"], gcnAUC["2.0"]}},
		{`Ring recovery over the same range goes from ([0-9.]+)% to ([0-9.]+)%`,
			[]float64{100 * gcnRecall["0.0"], 100 * gcnRecall["2.0"]}},
		{`even Integrated Gradients puts only ([0-9.]+)% of its top edges inside ` +
			`the ring, and the random-edge expectation for the same neighbourhoods ` +
			`is ([0-9.]+)%`,
			[]float64{100 * igPrecision, 100 * igNull}},
		{`moves from ([0-9.]+)x at camouflage 0 to ([0-9.]+)x at camouflage 2.0, and ` +
			`integrated gradients from ([0-9.]+)x to ([0-9.]+)x, peaking at ([0-9.]+)x ` +
			`at camouflage 1.0`,
			[]float64{gnnLift["0.0"], gnnLift["2.0"], igLift["0.0"], igLift["2.0"],
				igLift["1.0"]}},
		{`a clique is still ([0-9.]+)% recovered at camouflage 2.0, while the star ` +
			`is at zero by camouflage 1.0 and the cycle is down to ([0-9.]+)%`,
			[]float64{100 * gcnRingByCell["clique 2.0"], 100 * gcnRingByCell["cycle 1.0"],
				100 * gcnRingByCell["star 1.0"]}},
		{`structure-blind MLP on the same features scores ([0-9.]+) to ([0-9.]+) AUC ` +
			`by topology`, nil},
	}

	bad := 0
	fmt.Printf("\nnumbers written into the README prose\n")
	for _, c := range claims {
		re := regexp.MustCompile(c.pattern)
		m := re.FindStringSubmatch(prose)
		if m == nil {
			fmt.Printf("  claim not found in README.md: %s\n", c.pattern)
			bad++
			continue
		}
		got := c.got
		if got == nil { // the MLP range, whose ends are a min and a max
			lo, hi := math.Inf(1), math.Inf(-1)
			for _, v := range mlpAUC {
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
			got = []float64{lo, hi}
		}
		// The star claim carries a third value that must be exactly zero.
		if len(got) == len(m) { // one extra recomputed value, no capture for it
			if got[len(got)-1] != 0 {
				fmt.Printf("  the star is not at zero by camouflage 1.0: %.4f\n",
					got[len(got)-1])
				bad++
			}
			got = got[:len(got)-1]
		}
		for i, want := range m[1:] {
			if like(got[i], want) != want {
				fmt.Printf("  README says %s where the data gives %s (%.6f)\n",
					want, like(got[i], want), got[i])
				bad++
			}
		}
		fmt.Printf("  ok  %s\n", strings.Join(m[1:], ", "))
	}

	// The one claim that is a count rather than a mean.
	re := regexp.MustCompile(`sit above ([0-9.]+) AUC with ring recall below ` +
		`([0-9.]+), and across all three model families it is ([0-9]+) of ([0-9]+)`)
	m := re.FindStringSubmatch(prose)
	if m == nil {
		fmt.Printf("  the cell-count claim is not in README.md\n")
		return bad + 1
	}
	aucThreshold, _ := strconv.ParseFloat(m[1], 64)
	recallThreshold, _ := strconv.ParseFloat(m[2], 64)
	hits, total := 0, 0
	for k, a := range cellAUC {
		total++
		if round3(a) > aucThreshold && round3(cellRecall[k]) < recallThreshold {
			hits++
		}
	}
	if strconv.Itoa(hits) != m[3] || strconv.Itoa(total) != m[4] {
		fmt.Printf("  README says %s of %s cells above %s AUC and below %s recall, "+
			"the data gives %d of %d\n", m[3], m[4], m[1], m[2], hits, total)
		bad++
	} else {
		fmt.Printf("  ok  %s of %s cells above %s AUC with recall below %s\n",
			m[3], m[4], m[1], m[2])
	}
	return bad
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	bad := 0
	reports := filepath.Join(*root, "reports")

	// 1. Every results file under reports/, including the gzipped one.
	paths, _ := filepath.Glob(filepath.Join(reports, "*.csv"))
	gz, _ := filepath.Glob(filepath.Join(reports, "*.csv.gz"))
	paths = append(paths, gz...)
	sort.Strings(paths)
	if len(paths) == 0 {
		fmt.Fprintf(os.Stderr, "no results files under %s\n", reports)
		os.Exit(2)
	}

	loaded := map[string]*table{}
	fmt.Printf("validating %d results files under reports/\n", len(paths))
	for _, path := range paths {
		t, err := open(path)
		if err != nil {
			fmt.Printf("  %s: unreadable: %v\n", filepath.Base(path), err)
			bad++
			continue
		}
		loaded[filepath.Base(path)] = t
		for _, p := range validate(path, t) {
			fmt.Printf("  %s: %s\n", filepath.Base(path), p)
			bad++
		}
	}
	if bad == 0 {
		fmt.Printf("  no ragged rows, duplicate columns, empty cells, NaN or Inf anywhere\n")
	}

	det, okDet := loaded["detection_raw.csv"]
	fai, okFai := loaded["faithfulness_raw.csv.gz"]
	if !okDet || !okFai {
		fmt.Fprintln(os.Stderr, "the two raw tables are missing")
		os.Exit(2)
	}

	// 2. The flattened headers verify/summaries.sql imports have to be the
	// headers the published summaries actually carry.
	for _, pair := range [][2]string{
		{"detection_summary.csv", "det_summary_flat.csv"},
		{"faithfulness_summary.csv", "fai_summary_flat.csv"},
	} {
		pub, ok := loaded[pair[0]]
		if !ok {
			continue
		}
		flat, err := open(filepath.Join(*root, "verify", ".work", pair[1]))
		if err != nil {
			fmt.Printf("  %s: no flattened copy to compare (%v)\n", pair[1], err)
			bad++
			continue
		}
		want, got := composed(pub), flat.header
		if strings.Join(want, ",") != strings.Join(got, ",") {
			fmt.Printf("  %s: flattened header %v does not match %v\n", pair[0], got, want)
			bad++
		}
	}

	// 3. Row counts against what sweep_config.json says was run.
	raw, err := os.ReadFile(filepath.Join(reports, "sweep_config.json"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "sweep_config.json: %v\n", err)
		os.Exit(2)
	}
	var cfg struct {
		Topologies  []string  `json:"topologies"`
		Camouflage  []float64 `json:"camouflage"`
		Seeds       int       `json:"seeds"`
		NDetection  int       `json:"n_detection_rows"`
		NFaithfulls int       `json:"n_faithfulness_rows"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		fmt.Fprintf(os.Stderr, "sweep_config.json: %v\n", err)
		os.Exit(2)
	}

	distinct := func(t *table, name string) []string {
		seen := map[string]bool{}
		for _, r := range t.rows {
			seen[r[t.col(name)]] = true
		}
		out := make([]string, 0, len(seen))
		for k := range seen {
			out = append(out, k)
		}
		sort.Strings(out)
		return out
	}
	configs := map[string]bool{}
	for _, r := range det.rows {
		configs[r[det.col("topology")]+r[det.col("camouflage")]+r[det.col("seed")]] = true
	}

	// The CSV writes 0.0 and the JSON writes 0, so the levels are compared as
	// numbers rather than as the text either file happens to use.
	wantCamo := append([]float64(nil), cfg.Camouflage...)
	sort.Float64s(wantCamo)
	wantTopo := append([]string(nil), cfg.Topologies...)
	sort.Strings(wantTopo)

	type countCheck struct {
		what string
		got  int
		want int
	}
	checks := []countCheck{
		{"detection rows", len(det.rows), cfg.NDetection},
		{"faithfulness rows", len(fai.rows), cfg.NFaithfulls},
		{"seeds", len(distinct(det, "seed")), cfg.Seeds},
		{"topologies", len(distinct(det, "topology")), len(cfg.Topologies)},
		{"camouflage levels", len(distinct(det, "camouflage")), len(cfg.Camouflage)},
		{"configurations", len(configs), len(cfg.Topologies) * len(cfg.Camouflage) * cfg.Seeds},
		{"trained models", len(det.rows), len(configs) * len(distinct(det, "model"))},
		{"faithfulness measurements", len(fai.rows),
			len(distinct(fai, "topology")) * 0, // filled below
		},
	}
	// The last row is the product identity: cells x explainers x budgets x nodes
	// is not a fixed product because the node count varies, so it is dropped and
	// the explainer and budget counts are checked on their own instead.
	checks = checks[:len(checks)-1]
	checks = append(checks,
		countCheck{"explainers", len(distinct(fai, "explainer")), 4},
		countCheck{"explanation budgets", len(distinct(fai, "k_mode")), 6},
	)

	fmt.Printf("\ncounts against reports/sweep_config.json\n")
	for _, c := range checks {
		status := "ok"
		if c.got != c.want {
			status = "FAIL"
			bad++
		}
		fmt.Printf("  %-26s %8d  expected %8d  %s\n", c.what, c.got, c.want, status)
	}
	if strings.Join(distinct(det, "topology"), ",") != strings.Join(wantTopo, ",") {
		fmt.Printf("  topologies in the data do not match sweep_config.json\n")
		bad++
	}
	gotCamo := []float64{}
	for _, s := range distinct(det, "camouflage") {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			fmt.Printf("  camouflage %q is not a number\n", s)
			bad++
			continue
		}
		gotCamo = append(gotCamo, v)
	}
	sort.Float64s(gotCamo)
	if fmt.Sprint(gotCamo) != fmt.Sprint(wantCamo) {
		fmt.Printf("  camouflage levels %v do not match sweep_config.json %v\n",
			gotCamo, wantCamo)
		bad++
	}

	// 4. Table 1, recomputed from the two raw tables.
	detAcc := map[key]*acc{}
	for _, r := range det.rows {
		if r[det.col("model")] != "gcn" {
			continue
		}
		camo, _ := strconv.ParseFloat(r[det.col("camouflage")], 64)
		k := key{r[det.col("topology")], camo}
		if detAcc[k] == nil {
			detAcc[k] = &acc{}
		}
		detAcc[k].add([]float64{det.num(r, "auc"), det.num(r, "ring_recall")})
	}
	faiAcc := map[key]*acc{}
	for _, r := range fai.rows {
		if r[fai.col("model")] != "gcn" || r[fai.col("explainer")] != "gnnexplainer" ||
			r[fai.col("k_mode")] != "oracle" || r[fai.col("detected")] != "1" {
			continue
		}
		camo, _ := strconv.ParseFloat(r[fai.col("camouflage")], 64)
		k := key{r[fai.col("topology")], camo}
		if faiAcc[k] == nil {
			faiAcc[k] = &acc{}
		}
		faiAcc[k].add([]float64{fai.num(r, "precision"), fai.num(r, "random_expectation"),
			fai.num(r, "lift"), fai.num(r, "n_candidates")})
	}

	pubTables, err := mdTables(filepath.Join(reports, "tables.md"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "tables.md: %v\n", err)
		os.Exit(2)
	}
	if len(pubTables) == 0 {
		fmt.Fprintln(os.Stderr, "no tables in reports/tables.md")
		os.Exit(2)
	}
	t1 := pubTables[0]
	if len(t1) != 17 || len(t1[0]) != 9 {
		fmt.Fprintf(os.Stderr, "table 1 is %d rows by %d columns, expected 17 by 9\n",
			len(t1), len(t1[0]))
		os.Exit(2)
	}

	fmt.Printf("\ntable 1, recomputed from detection_raw.csv and faithfulness_raw.csv.gz\n")
	worst := 0.0
	for _, row := range t1[1:] {
		camo, _ := strconv.ParseFloat(row[1], 64)
		k := key{row[0], camo}
		d, okD := detAcc[k]
		fv, okF := faiAcc[k]
		if !okD || !okF {
			fmt.Printf("  %-10s %-4s no rows in the raw data\n", row[0], row[1])
			bad++
			continue
		}
		got := []float64{d.mean(0), d.mean(1), fv.mean(0), fv.mean(1), fv.mean(2),
			fv.mean(3), float64(fv.n)}
		rowBad := false
		for i, g := range got {
			want, _ := strconv.ParseFloat(row[i+2], 64)
			tol := tableTol
			if i == 6 {
				tol = 0 // nodes explained is a count
			}
			delta := math.Abs(g - want)
			if delta > worst {
				worst = delta
			}
			if delta > tol {
				rowBad = true
			}
		}
		if rowBad {
			bad++
		}
		fmt.Printf("  %-10s camo %-4s auc %.4f  recall %.4f  precision %.4f  "+
			"null %.4f  lift %.4f  cand %.3f  n %d  %s\n",
			row[0], row[1], got[0], got[1], got[2], got[3], got[4], got[5], int(got[6]),
			map[bool]string{true: "FAIL", false: "ok"}[rowBad])
	}
	fmt.Printf("  worst |recomputed - published| across all 16 rows: %.3e\n", worst)

	// 5. The README is a paste of reports/tables.md. Nothing checked that.
	readme, err := mdTables(filepath.Join(*root, "README.md"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "README.md: %v\n", err)
		os.Exit(2)
	}
	fmt.Printf("\n%d generated tables against the copies pasted into README.md\n", len(pubTables))
	missing := 0
	for i, want := range pubTables {
		found := false
		for _, got := range readme {
			if sameTable(want, got) {
				found = true
				break
			}
		}
		if !found {
			fmt.Printf("  table %d is not in README.md, or differs from the generated one\n", i+1)
			missing++
			bad++
		}
	}
	if missing == 0 {
		fmt.Printf("  all %d appear in README.md cell for cell\n", len(pubTables))
	}

	// 6. The numbers written into the prose. Every table above is generated,
	// but the sentences a reader actually remembers were typed by hand, and
	// nothing checked them against the data at all.
	bad += checkProse(*root, det, fai)

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Printf("\nGo agrees with Table 1 and with the prose, and reports/ is well formed\n")
}
