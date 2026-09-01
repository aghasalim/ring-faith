//! Is the analytic random-edge null right?
//!
//! Every faithfulness number in this repository is read against one number:
//! `random_expectation`, the precision an explainer would get by picking edges
//! at random from the same candidate set. `src/ringfaith/metrics.py` writes it
//! as `n_relevant / n_candidates`, one line, closed form. The check offered on
//! it is that a random explainer lands on lift 1.0, which is an aggregate over
//! four thousand nodes and would not notice a null that is wrong in a way that
//! cancels.
//!
//! This simulates it instead. For each of the 131136 measured rows, drawing k
//! of the n_candidates edges uniformly without replacement gives a
//! hypergeometric number of hits, so this draws exactly that, many times, and
//! compares the simulated mean precision against the closed form. Python could
//! not afford it: 131136 rows times 500 draws is a few hundred million
//! samples.
//!
//! It also checks the repository's own control the same way. The random
//! explainer's measured precision is one draw per node; the simulation is 500,
//! so the two are compared against the same analytic value with the simulation
//! as the reference.
//!
//! Usage: cargo run --release -- <repo root>

use std::env;
use std::fs;
use std::process::exit;

const DRAWS: usize = 500;
const SIGMA: f64 = 5.0;

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast, and seeded reproducibly so a failure here can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        // 53 bits of mantissa, which is every double in [0, 1) that matters.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

struct Row {
    explainer: String,
    k_mode: String,
    detected: u8,
    precision: f64,
    random_expectation: f64,
    n_candidates: u64,
    n_relevant: u64,
    k: u64,
}

/// One draw of k edges out of n_candidates without replacement, counting how
/// many are ring edges. Sampling the count directly rather than shuffling the
/// candidate set is what makes this O(k) instead of O(n_candidates).
fn hypergeometric(n: u64, relevant: u64, k: u64, rng: &mut Rng) -> u64 {
    let mut left = n;
    let mut good = relevant;
    let mut hits = 0;
    for _ in 0..k {
        if rng.unit() < good as f64 / left as f64 {
            hits += 1;
            good -= 1;
        }
        left -= 1;
    }
    hits
}

fn load(root: &str) -> Vec<Row> {
    let path = format!("{}/verify/.work/faithfulness_raw.csv", root);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        eprintln!("run verify/verify.sh, which unpacks the gzipped table");
        exit(2)
    });

    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("empty file").split(',').collect();
    let col = |name: &str| {
        header.iter().position(|h| *h == name).unwrap_or_else(|| {
            eprintln!("no column {} in faithfulness_raw.csv", name);
            exit(2)
        })
    };
    let (c_ex, c_km, c_det) = (col("explainer"), col("k_mode"), col("detected"));
    let c_prec = col("precision");
    let c_exp = col("random_expectation");
    let (c_cand, c_rel, c_k) = (col("n_candidates"), col("n_relevant"), col("k"));

    let mut rows = Vec::with_capacity(131_136);
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let parse = |i: usize| -> f64 {
            f[i].trim().parse().unwrap_or_else(|_| {
                eprintln!("cannot parse {:?}", f[i]);
                exit(2)
            })
        };
        rows.push(Row {
            explainer: f[c_ex].to_string(),
            k_mode: f[c_km].to_string(),
            detected: f[c_det].trim().parse().unwrap_or(0),
            precision: parse(c_prec),
            random_expectation: parse(c_exp),
            n_candidates: parse(c_cand) as u64,
            n_relevant: parse(c_rel) as u64,
            k: parse(c_k) as u64,
        });
    }
    rows
}

#[derive(Default)]
struct Group {
    n: usize,
    analytic: f64,
    simulated: f64,
    observed: f64,
    variance: f64, // summed variance of the simulated per-row means
    observed_var: f64,
}

fn summarise(rows: &[Row], select: &dyn Fn(&Row) -> Option<String>) -> Vec<(String, Group)> {
    let mut keys: Vec<String> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();

    for (i, r) in rows.iter().enumerate() {
        let key = match select(r) {
            Some(k) => k,
            None => continue,
        };
        if r.k == 0 || r.n_candidates == 0 {
            continue;
        }
        let idx = match keys.iter().position(|k| *k == key) {
            Some(j) => j,
            None => {
                keys.push(key);
                groups.push(Group::default());
                groups.len() - 1
            }
        };

        let mut rng = Rng::new(0x5EED_0000_0000 + i as u64 * 6_364_136_223_846_793_005);
        let mut hits = 0u64;
        for _ in 0..DRAWS {
            hits += hypergeometric(r.n_candidates, r.n_relevant, r.k, &mut rng);
        }
        let sim = hits as f64 / (DRAWS as f64 * r.k as f64);

        // Variance of one draw's precision, hypergeometric over k picks.
        let p = r.n_relevant as f64 / r.n_candidates as f64;
        let n = r.n_candidates as f64;
        let k = r.k as f64;
        let var_one = if n > 1.0 {
            p * (1.0 - p) * (n - k) / (n - 1.0) / k
        } else {
            0.0
        };

        let g = &mut groups[idx];
        g.n += 1;
        g.analytic += r.random_expectation;
        g.simulated += sim;
        g.observed += r.precision;
        g.variance += var_one / DRAWS as f64;
        g.observed_var += var_one;
    }
    keys.into_iter().zip(groups).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");
    let rows = load(root);
    println!("{} measured rows, {} draws each", rows.len(), DRAWS);
    println!(
        "sampling k of n_candidates without replacement and comparing the mean\n\
         simulated precision against n_relevant / n_candidates\n"
    );

    let mut failures = 0;

    println!("every row, grouped by explanation budget");
    println!(
        "  {:<8} {:>7}  {:>9}  {:>9}  {:>8}  {:>6}",
        "budget", "rows", "analytic", "simulated", "sd", "z"
    );
    let by_budget = summarise(&rows, &|r| Some(r.k_mode.clone()));
    for (key, g) in &by_budget {
        let n = g.n as f64;
        let analytic = g.analytic / n;
        let simulated = g.simulated / n;
        let sd = g.variance.sqrt() / n;
        let z = (simulated - analytic) / sd.max(1e-15);
        let ok = z.abs() <= SIGMA;
        if !ok {
            failures += 1;
        }
        println!(
            "  {:<8} {:>7} {:>10.6} {:>10.6} {:>9.2e} {:>+7.2}  {}",
            key,
            g.n,
            analytic,
            simulated,
            sd,
            z,
            if ok { "ok" } else { "FAIL" }
        );
    }

    // The repository's own control, checked the same way. Its measured
    // precision is one draw per node, so the comparison is noisier by
    // sqrt(DRAWS) and is scored against its own standard error.
    println!("\nthe random explainer against the same simulation, detected nodes");
    println!(
        "  {:<8} {:>7}  {:>9}  {:>9}  {:>9}  {:>6}",
        "budget", "rows", "analytic", "simulated", "measured", "z"
    );
    let control = summarise(&rows, &|r| {
        if r.explainer == "random" && r.detected == 1 {
            Some(r.k_mode.clone())
        } else {
            None
        }
    });
    for (key, g) in &control {
        let n = g.n as f64;
        let analytic = g.analytic / n;
        let simulated = g.simulated / n;
        let observed = g.observed / n;
        let sd = g.observed_var.sqrt() / n;
        let z = (observed - simulated) / sd.max(1e-15);
        let ok = z.abs() <= SIGMA;
        if !ok {
            failures += 1;
        }
        println!(
            "  {:<8} {:>7} {:>10.6} {:>10.6} {:>10.6} {:>+7.2}  {}",
            key,
            g.n,
            analytic,
            simulated,
            observed,
            z,
            if ok { "ok" } else { "FAIL" }
        );
    }

    if failures > 0 {
        println!(
            "\n{} groups sit further than {} standard errors from the simulation",
            failures, SIGMA
        );
        exit(1);
    }
    println!(
        "\nthe closed-form null matches a {}-draw simulation on every budget, and the\n\
         random explainer lands on it within {} standard errors",
        DRAWS, SIGMA
    );
}
