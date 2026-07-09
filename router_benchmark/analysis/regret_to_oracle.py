#!/usr/bin/env python3
"""Regret-to-oracle: for each real router, how much worse (in cost, at
matched or better success) is it than the oracle upper bound on the same
tasks. Regret per task: both succeed -> router_cost - oracle_cost (>=0);
router fails, oracle succeeds -> oracle_cost + benchmark-group mean
successful-task cost (a fixed failure penalty, since a failed task has no
well-defined "cost of the miss" otherwise); router succeeds, oracle
doesn't (should not happen by construction, guarded anyway) -> 0."""
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
    oracle_path = Path(__file__).parent / "output" / "oracle_and_cascade.csv"
    with open(oracle_path) as f:
        oracle = {(r["benchmark_name"], r["task_id"]): r for r in csv.DictReader(f)}

    results_path = Path(__file__).parent / "../output/live/paper1_live_v2/results.csv"
    with open(results_path) as f:
        results = list(csv.DictReader(f))

    group_success_costs = defaultdict(list)
    for r in results:
        if r["success"] == "True":
            group_success_costs[group_key(r["benchmark_name"])].append(float(r["cost_usd"]))
    mean_success_cost = {g: sum(v) / len(v) for g, v in group_success_costs.items() if v}

    regrets = defaultdict(list)
    for r in results:
        key = (r["benchmark_name"], r["task_id"])
        if key not in oracle:
            continue
        o = oracle[key]
        group = group_key(r["benchmark_name"])
        router_success = r["success"] == "True"
        oracle_success = o["oracle_success"] == "True"
        if router_success and oracle_success:
            regret = float(r["cost_usd"]) - float(o["oracle_cost"])
        elif (not router_success) and oracle_success:
            regret = float(o["oracle_cost"]) + mean_success_cost.get(group, 0.0)
        else:
            regret = 0.0
        regrets[(r["router_name"], group)].append(regret)

    out_path = Path(__file__).parent / "output" / "regret_to_oracle.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "benchmark_group", "mean_regret", "n_tasks"])
        for (router, group), vals in sorted(regrets.items()):
            w.writerow([router, group, f"{sum(vals)/len(vals):.6f}", len(vals)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
