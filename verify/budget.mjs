// Recompute Tables 4, 8 and 9 from the per-node raw table, in Node.
//
// Table 8 and Table 9 are the budget sweep: mean lift over the analytic null
// for each explainer at each explanation budget, on the fraud nodes the model
// caught and on the ones it missed. They are the evidence for the claim that
// the oracle budget flattered the plain gradient, and they were produced by one
// pandas pivot. Table 4 is the dissociation itself, the ordering of topologies
// by node AUC against their ordering by lift, which is the finding the repo is
// named for.
//
// Usage: node verify/budget.mjs <repo root>

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? ".";
const TOL = 6e-4; // published to 3 decimals, so half a step is 5e-4
const BUDGETS = ["k1", "k3", "k5", "k10", "k20", "oracle"];

function readCsv(path) {
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  const header = lines[0].split(",");
  const at = Object.fromEntries(header.map((h, i) => [h, i]));
  for (const name of ["topology", "camouflage", "model", "explainer", "k_mode",
                      "detected", "precision", "random_expectation", "lift",
                      "n_candidates", "auc", "ring_recall"]) {
    void name;
  }
  return { at, rows: lines.slice(1).map((l) => l.split(",")) };
}

// One table out of reports/tables.md, found by the heading it sits under.
function mdTable(text, heading) {
  const lines = text.split("\n");
  const start = lines.findIndex((l) => l.startsWith(heading));
  if (start < 0) throw new Error(`no heading ${heading} in tables.md`);
  const out = [];
  for (let i = start + 1; i < lines.length; i++) {
    const s = lines[i].trim();
    if (!s.startsWith("|")) {
      if (out.length > 0) break;
      continue;
    }
    const cells = s.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    if (cells.join("").replace(/[-:]/g, "") === "") continue;
    out.push(cells);
  }
  if (out.length < 2) throw new Error(`no table under ${heading}`);
  return { header: out[0], rows: out.slice(1) };
}

const round3 = (x) => Math.round(x * 1000) / 1000;

function meanBy(rows, keyFn, valueFns) {
  const acc = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    if (k === null) continue;
    let a = acc.get(k);
    if (!a) {
      a = { sum: valueFns.map(() => 0), n: 0 };
      acc.set(k, a);
    }
    valueFns.forEach((f, i) => (a.sum[i] += f(r)));
    a.n++;
  }
  return acc;
}

const tables = readFileSync(join(root, "reports", "tables.md"), "utf8");
const det = readCsv(join(root, "reports", "detection_raw.csv"));
let fai;
try {
  fai = readCsv(join(root, "verify", ".work", "faithfulness_raw.csv"));
} catch (e) {
  console.error(`${e.message}\nrun verify/verify.sh, which unpacks the gzipped table`);
  process.exit(2);
}

let failures = 0;
const report = (label, got, want, tol) => {
  const bad = Math.abs(got - want) > tol;
  if (bad) failures++;
  return { bad, delta: Math.abs(got - want), text:
    `${label} recomputed ${got.toFixed(4)} published ${want} ${bad ? "FAIL" : "ok"}` };
};

// ---------------------------------------------------------------- tables 8, 9
let worst = 0;
for (const [n, detected] of [[8, "1"], [9, "0"]]) {
  const pub = mdTable(tables, `### Table ${n}.`);
  const acc = meanBy(
    fai.rows.filter((r) => r[fai.at.detected] === detected),
    (r) => `${r[fai.at.explainer]}|${r[fai.at.k_mode]}`,
    [(r) => Number(r[fai.at.lift])],
  );
  console.log(`\ntable ${n}, mean lift by budget on detected=${detected} nodes`);
  for (const row of pub.rows) {
    const explainer = row[0];
    const cells = [];
    let rowBad = false;
    for (let i = 0; i < BUDGETS.length; i++) {
      const a = acc.get(`${explainer}|${BUDGETS[i]}`);
      if (!a) {
        console.log(`  ${explainer}: no ${BUDGETS[i]} rows in the raw data`);
        failures++;
        rowBad = true;
        continue;
      }
      const got = a.sum[0] / a.n;
      const want = Number(row[i + 1]);
      const delta = Math.abs(got - want);
      worst = Math.max(worst, delta);
      if (delta > TOL) rowBad = true;
      cells.push(got.toFixed(4));
    }
    if (rowBad) failures++;
    console.log(`  ${explainer.padEnd(13)} ${cells.join("  ")}  ${rowBad ? "FAIL" : "ok"}`);
  }
}

// ------------------------------------------------------------------- table 4
// Table 4 averages the Table 1 cells, and make_tables.py rounds Table 1 to
// three decimals before averaging, so the same rounding is applied here.
const detAcc = meanBy(
  det.rows.filter((r) => r[det.at.model] === "gcn"),
  (r) => `${r[det.at.topology]}|${Number(r[det.at.camouflage])}`,
  [(r) => Number(r[det.at.auc])],
);
const faiAcc = meanBy(
  fai.rows.filter((r) => r[fai.at.model] === "gcn" && r[fai.at.explainer] === "gnnexplainer"
    && r[fai.at.k_mode] === "oracle" && r[fai.at.detected] === "1"),
  (r) => `${r[fai.at.topology]}|${Number(r[fai.at.camouflage])}`,
  [(r) => Number(r[fai.at.precision]), (r) => Number(r[fai.at.lift])],
);

const byTopology = new Map();
for (const [k, a] of faiAcc) {
  const [topology] = k.split("|");
  const d = detAcc.get(k);
  if (!d) throw new Error(`no detection rows for ${k}`);
  let t = byTopology.get(topology);
  if (!t) {
    t = { auc: 0, precision: 0, lift: 0, n: 0 };
    byTopology.set(topology, t);
  }
  t.auc += round3(d.sum[0] / d.n);
  t.precision += round3(a.sum[0] / a.n);
  t.lift += round3(a.sum[1] / a.n);
  t.n++;
}

const diss = [...byTopology.entries()].map(([topology, t]) => ({
  topology,
  auc: t.auc / t.n,
  precision: t.precision / t.n,
  lift: t.lift / t.n,
}));
const rankOf = (arr, field) => {
  const order = [...arr].sort((a, b) => b[field] - a[field]);
  return Object.fromEntries(order.map((d, i) => [d.topology, i + 1]));
};
const aucRank = rankOf(diss, "auc");
const liftRank = rankOf(diss, "lift");

const pub4 = mdTable(tables, "### Table 4.");
console.log("\ntable 4, the dissociation, averaged over camouflage");
for (const row of pub4.rows) {
  const d = diss.find((x) => x.topology === row[0]);
  if (!d) {
    console.log(`  ${row[0]}: not in the raw data`);
    failures++;
    continue;
  }
  const checks = [
    report("auc", d.auc, Number(row[1]), TOL),
    report("precision", d.precision, Number(row[2]), TOL),
    report("lift", d.lift, Number(row[3]), TOL),
    report("auc rank", aucRank[d.topology], Number(row[4]), 0),
    report("lift rank", liftRank[d.topology], Number(row[5]), 0),
  ];
  for (const c of checks) worst = Math.max(worst, c.delta);
  const bad = checks.some((c) => c.bad);
  console.log(`  ${row[0].padEnd(10)} auc ${d.auc.toFixed(4)}  precision ` +
    `${d.precision.toFixed(4)}  lift ${d.lift.toFixed(4)}  ranks ` +
    `${aucRank[d.topology]}/${liftRank[d.topology]}  ${bad ? "FAIL" : "ok"}`);
}

console.log(`\nworst |recomputed - published| across tables 4, 8 and 9: ${worst.toExponential(3)}`);
if (failures > 0) {
  console.log(`${failures} disagreements`);
  process.exit(1);
}
console.log("Node reproduces the budget sweep and the dissociation ranking");
