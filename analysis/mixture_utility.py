#!/usr/bin/env python3
"""Expected success rate per router under named benchmark-group mixtures.
Existing-data only: weighted combination of already-computed per-group
success_rate / cost_per_task point estimates from bootstrap_ci.csv. This
reweights success only; a cost- and latency-aware utility is in
expected_utility.py."""
import csv
from collections import defaultdict
from pathlib import Path

MIXTURES = {
    "100pct_RouterBench": {"RouterBench": 1.0},
    "50_25_25_RB_BFCL_tau2": {"RouterBench": 0.5, "BFCL": 0.25, "tau2-bench": 0.25},
    "uniform": {"RouterBench": 0.25, "BFCL": 0.25, "tau2-bench": 0.25, "WebArena": 0.25},
    "WebArena_heavy": {"RouterBench": 0.1, "BFCL": 0.1, "tau2-bench": 0.1, "WebArena": 0.7},
}


def main():
    ci_path = Path(__file__).parent / "output" / "bootstrap_ci.csv"
    with open(ci_path) as f:
        rows = list(csv.DictReader(f))

    point = defaultdict(dict)
    for r in rows:
        point[(r["router"], r["benchmark_group"])][r["metric"]] = float(r["point_estimate"])

    groups = sorted({g for (_, g) in point.keys()})
    routers = sorted({rt for (rt, _) in point.keys()})
    routers = [rt for rt in routers if all((rt, g) in point for g in groups)]

    mean_cost = {}
    for g in groups:
        vals = [point[(rt, g)]["cost_per_task"] for rt in routers]
        mean_cost[g] = sum(vals) / len(vals)
    inv = {g: 1 / mean_cost[g] for g in groups}
    total_inv = sum(inv.values())
    MIXTURES["cost_constrained"] = {g: inv[g] / total_inv for g in groups}

    out_path = Path(__file__).parent / "output" / "mixture_utility.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "mixture", "expected_success_rate", "expected_cost_per_task"])
        for router in routers:
            for mix_name, weights in MIXTURES.items():
                dist_utility = sum(w_ * point[(router, g)]["success_rate"] for g, w_ in weights.items())
                exp_cost = sum(w_ * point[(router, g)]["cost_per_task"] for g, w_ in weights.items())
                w.writerow([router, mix_name, f"{dist_utility:.4f}", f"{exp_cost:.6f}"])
    print(f"wrote {out_path}")
    print("cost_constrained weights:", MIXTURES["cost_constrained"])


if __name__ == "__main__":
    main()
