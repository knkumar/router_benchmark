#!/usr/bin/env python3
"""Cost normalized to the strong-frontier candidate's real per-task cost,
so the Pareto-frontier shape survives future price changes even though
the raw-dollar tables (tied to PRICING_ASOF 2026-07-02) will not.
Reads sweep_results.csv (Task 1) for the strong-frontier baseline cost
per task, and paper1_live_v2/results.csv for each router's real cost."""
import csv
from collections import defaultdict
from pathlib import Path


def group_key(benchmark_name):
    if "RouterBench" in benchmark_name:
        return "RouterBench"
    if "BFCL" in benchmark_name:
        return "BFCL"
    if "tau2" in benchmark_name:
        return "tau2-bench"
    if "WebArena" in benchmark_name:
        return "WebArena"
    raise ValueError(benchmark_name)


def main():
    sweep_path = Path(__file__).parent / "../output/live/sweep_v1/sweep_results.csv"
    with open(sweep_path) as f:
        sweep = list(csv.DictReader(f))
    strong_cost = {
        (r["benchmark_name"], r["task_id"]): float(r["cost_usd"])
        for r in sweep if r["candidate_tier"] == "strong-frontier"
    }

    results_path = Path(__file__).parent / "../output/live/paper1_live_v2/results.csv"
    with open(results_path) as f:
        results = list(csv.DictReader(f))

    ratios = defaultdict(list)
    for r in results:
        key = (r["benchmark_name"], r["task_id"])
        if key not in strong_cost or strong_cost[key] == 0:
            continue
        ratio = float(r["cost_usd"]) / strong_cost[key]
        ratios[(r["router_name"], group_key(r["benchmark_name"]))].append(ratio)

    out_path = Path(__file__).parent / "output" / "normalized_cost.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "benchmark_group", "mean_cost_ratio_to_strong_frontier", "n_tasks"])
        for (router, group), vals in sorted(ratios.items()):
            w.writerow([router, group, f"{sum(vals)/len(vals):.4f}", len(vals)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
