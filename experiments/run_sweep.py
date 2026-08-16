"""Core experiment: ring topology x camouflage, node AUC vs explainer faithfulness.

    python experiments/run_sweep.py --seeds 5 --out reports
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from ringfaith.experiment import run_config
from ringfaith.generate import TOPOLOGIES

CAMOUFLAGE = (0.0, 0.5, 1.0, 2.0)


def _cell(args):
    """One (topology, camouflage, seed) cell, pinned to one thread.

    Every cell reseeds torch at entry to `run_config`, so cells are independent and the
    result does not depend on how they are scheduled. One thread per worker is not just for
    packing: matmul reduction order depends on the thread count, so pinning it is what makes
    a parallel run reproduce a serial one bit for bit.
    """
    topology, camouflage, seed, n_explain = args
    torch.set_num_threads(1)
    return run_config(topology, camouflage, seed, n_explain=n_explain)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-explain", type=int, default=25)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--topologies", nargs="*", default=list(TOPOLOGIES))
    p.add_argument("--camouflage", nargs="*", type=float, default=list(CAMOUFLAGE))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cells = [
        (t, c, s, a.n_explain) for t in a.topologies for c in a.camouflage for s in range(a.seeds)
    ]
    det, faith = [], []
    t0 = time.time()
    # `map` yields in submission order, so the written rows do not depend on the worker count.
    # spawn rather than the Linux default fork: forking a process that has already started
    # torch's OpenMP pool is a known way to deadlock, and a hung CI job is worse than a
    # slightly slower start.
    with ProcessPoolExecutor(
        max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for i, ((topo, camo, seed, _), (d, f)) in enumerate(zip(cells, pool.map(_cell, cells))):
            det += d
            faith += f
            print(
                f"[{i + 1}/{len(cells)}] {topo:10s} camo={camo:<4} seed={seed} "
                f"({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )

    det_df, faith_df = pd.DataFrame(det), pd.DataFrame(faith)
    det_df.to_csv(a.out / "detection_raw.csv", index=False)
    # 131k rows of mostly repeated columns: 15 MB raw, 1.1 MB gzipped, and pandas reads
    # the compression straight off the extension. No reason to commit the raw one.
    faith_df.to_csv(a.out / "faithfulness_raw.csv.gz", index=False)

    det_s = (
        det_df.groupby(["topology", "camouflage", "model"])[["auc", "ap", "ring_recall"]]
        .agg(["mean", "std"])
        .round(4)
    )
    faith_s = (
        faith_df.groupby(
            ["topology", "camouflage", "model", "explainer", "detected", "k_mode"]
        )[["precision", "random_expectation", "lift", "n_candidates", "n_relevant"]]
        .agg(["mean", "std"])
        .round(4)
    )
    det_s.to_csv(a.out / "detection_summary.csv")
    faith_s.to_csv(a.out / "faithfulness_summary.csv")

    (a.out / "sweep_config.json").write_text(
        json.dumps(
            {
                "topologies": a.topologies,
                "camouflage": a.camouflage,
                "seeds": a.seeds,
                "n_explain": a.n_explain,
                "n_detection_rows": len(det_df),
                "n_faithfulness_rows": len(faith_df),
                "runtime_seconds": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.out}")
    print(det_s.to_string())
    print(faith_s.to_string())


if __name__ == "__main__":
    main()
