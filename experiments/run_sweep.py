"""Core experiment: ring topology x camouflage, node AUC vs explainer faithfulness.

    python experiments/run_sweep.py --seeds 5 --out reports
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ringfaith.experiment import run_config
from ringfaith.generate import TOPOLOGIES

CAMOUFLAGE = (0.0, 0.5, 1.0, 2.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-explain", type=int, default=25)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--topologies", nargs="*", default=list(TOPOLOGIES))
    p.add_argument("--camouflage", nargs="*", type=float, default=list(CAMOUFLAGE))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    det, faith = [], []
    total = len(a.topologies) * len(a.camouflage) * a.seeds
    t0 = time.time()
    for i, (topo, camo, seed) in enumerate(
        (t, c, s) for t in a.topologies for c in a.camouflage for s in range(a.seeds)
    ):
        d, f = run_config(topo, camo, seed, n_explain=a.n_explain)
        det += d
        faith += f
        print(
            f"[{i + 1}/{total}] {topo:10s} camo={camo:<4} seed={seed} "
            f"({time.time() - t0:.0f}s elapsed)",
            flush=True,
        )

    det_df, faith_df = pd.DataFrame(det), pd.DataFrame(faith)
    det_df.to_csv(a.out / "detection_raw.csv", index=False)
    faith_df.to_csv(a.out / "faithfulness_raw.csv", index=False)

    det_s = (
        det_df.groupby(["topology", "camouflage", "model"])[["auc", "ap", "ring_recall"]]
        .agg(["mean", "std"])
        .round(4)
    )
    faith_s = (
        faith_df.groupby(["topology", "camouflage", "model", "explainer"])[
            ["precision", "random_expectation", "lift", "n_candidates", "n_relevant"]
        ]
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
