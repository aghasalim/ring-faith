# Recompute Table 3 and Table 7 from the raw tables, in Ruby.
#
# Table 3 is the floor the whole repository rests on: a structure-blind MLP on
# the same features, so that whatever a graph model scores above it is what the
# structure bought. Table 7 is the detected-against-missed split, the table that
# answers what an explanation looks like on the fraud the model did not catch.
# Both came out of one pandas groupby each.
#
# The Wilcoxon p in the last column of Table 7 is not checked here; that column
# is checked in verify/verify.R, which has the distributions. Everything in
# Table 7 that is a mean or a count is checked here.
#
# Usage: ruby verify/tables37.rb <repo root>

root = ARGV[0] || "."
TOL = 6e-4          # both tables are published to 3 decimals
CELL = %w[topology camouflage seed model].freeze

def read_csv(path)
  lines = File.readlines(path, chomp: true).reject(&:empty?)
  header = lines.shift.split(",")
  at = {}
  header.each_with_index { |h, i| at[h] = i }
  [at, lines.map { |l| l.split(",") }]
end

# One table out of reports/tables.md, found by the heading above it.
def md_table(text, heading)
  lines = text.split("\n")
  start = lines.index { |l| l.start_with?(heading) }
  raise "no heading #{heading} in tables.md" if start.nil?

  out = []
  lines[(start + 1)..].each do |line|
    s = line.strip
    unless s.start_with?("|")
      break unless out.empty?

      next
    end
    cells = s.sub(/\A\|/, "").sub(/\|\z/, "").split("|").map(&:strip)
    next if cells.join.gsub(/[-:]/, "").empty?

    out << cells
  end
  raise "no table under #{heading}" if out.size < 2

  { header: out[0], rows: out[1..] }
end

def mean(values)
  values.sum / values.size.to_f
end

tables = File.read(File.join(root, "reports", "tables.md"))
det_at, det_rows = read_csv(File.join(root, "reports", "detection_raw.csv"))
fai_path = File.join(root, "verify", ".work", "faithfulness_raw.csv")
unless File.exist?(fai_path)
  warn "cannot open #{fai_path}"
  warn "run verify/verify.sh, which unpacks the gzipped table"
  exit 2
end
fai_at, fai_rows = read_csv(fai_path)

failures = 0
worst = 0.0

check = lambda do |got, want, tol|
  delta = (got - want).abs
  worst = delta if delta > worst
  bad = delta > tol
  failures += 1 if bad
  bad
end

# ------------------------------------------------------------------- table 3
pub3 = md_table(tables, "### Table 3.")
groups = Hash.new { |h, k| h[k] = { auc: [], rr: [] } }
det_rows.each do |r|
  g = groups[[r[det_at["topology"]], r[det_at["model"]]]]
  g[:auc] << r[det_at["auc"]].to_f
  g[:rr] << r[det_at["ring_recall"]].to_f
end

# The column headings carry the model, so which cell means what is read off the
# published table rather than assumed from its position.
columns = pub3[:header][1..].map do |h|
  m = h.match(/\A(\w+) \((\w+)\)\z/)
  raise "cannot read the model out of column #{h.inspect}" if m.nil?

  [m[1], m[2]]
end

puts "table 3, the structure-blind baseline"
pub3[:rows].each do |row|
  topology = row[0]
  bad = false
  shown = []
  columns.each_with_index do |(metric, model), i|
    g = groups[[topology, model]]
    if g[:auc].empty?
      puts "  #{topology}/#{model}: no rows in detection_raw.csv"
      failures += 1
      bad = true
      next
    end
    got = metric == "auc" ? mean(g[:auc]) : mean(g[:rr])
    bad = true if check.call(got, row[i + 1].to_f, TOL)
    shown << format("%.4f", got)
  end
  puts format("  %-10s %s  %s", topology, shown.join("  "), bad ? "FAIL" : "ok")
end

# ------------------------------------------------------------------- table 7
pub7 = md_table(tables, "### Table 7.")
oracle = fai_rows.select { |r| r[fai_at["k_mode"]] == "oracle" }
cell_cols = CELL.map { |c| fai_at[c] }

stats = Hash.new do |h, k|
  h[k] = { 1 => { lift: [], precision: [], cells: {} },
           0 => { lift: [], precision: [], cells: {} } }
end
oracle.each do |r|
  s = stats[[r[fai_at["model"]], r[fai_at["explainer"]]]][r[fai_at["detected"]].to_i]
  s[:lift] << r[fai_at["lift"]].to_f
  s[:precision] << r[fai_at["precision"]].to_f
  s[:cells][cell_cols.map { |c| r[c] }.join("|")] = true
end

puts "\ntable 7, faithfulness on detected against missed fraud nodes"
pub7[:rows].each do |row|
  model, explainer = row[0], row[1]
  s = stats[[model, explainer]]
  if s[1][:lift].empty? || s[0][:lift].empty?
    puts "  #{model}/#{explainer}: missing one side of the split"
    failures += 1
    next
  end
  hit_lift = mean(s[1][:lift])
  miss_lift = mean(s[0][:lift])
  paired = (s[1][:cells].keys & s[0][:cells].keys).size

  bad = false
  bad = true if check.call(hit_lift, row[2].to_f, TOL)
  bad = true if check.call(mean(s[1][:precision]), row[3].to_f, TOL)
  bad = true if check.call(s[1][:lift].size, row[4].to_i, 0)
  bad = true if check.call(miss_lift, row[5].to_f, TOL)
  bad = true if check.call(mean(s[0][:precision]), row[6].to_f, TOL)
  bad = true if check.call(s[0][:lift].size, row[7].to_i, 0)
  bad = true if check.call(hit_lift - miss_lift, row[8].to_f, TOL)
  bad = true if check.call(paired, row[9].to_i, 0)

  puts format("  %-5s %-13s detected %.4f/%d  missed %.4f/%d  paired cells %d  %s",
              model, explainer, hit_lift, s[1][:lift].size, miss_lift,
              s[0][:lift].size, paired, bad ? "FAIL" : "ok")
end

puts format("\nworst |recomputed - published| across tables 3 and 7: %.3e", worst)
if failures.positive?
  puts "#{failures} disagreements"
  exit 1
end
puts "Ruby reproduces the structure-blind baseline and the detected/missed split"
