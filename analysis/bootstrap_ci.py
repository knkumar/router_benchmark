#!/usr/bin/env python3
"""Bootstrap 95% CIs over tasks for each router x benchmark-group x metric.
Reads paper1_live_v3/results.csv (the full merge); writes bootstrap_ci.csv.
No new experiment data is generated -- this only resamples existing rows.
"""
import csv
import random
from pathlib import Path

random.seed(20260705)  # fixed seed for reproducibility

SOURCE = "../output/results.csv"
N_BOOT = 10000


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def group_key(benchmark_name):
    if "RouterBench" in benchmark_name:
        return "RouterBench"
    if "BFCL" in benchmark_name:
        return "BFCL"
    if "tau2" in benchmark_name:
        return "tau2-bench"
    if "WebArena" in benchmark_name:
        return "WebArena"
    return benchmark_name


def success_rate(rows):
    return sum(r["success"] == "True" for r in rows) / len(rows)


def cost_per_task(rows):
    return sum(float(r["cost_usd"]) for r in rows) / len(rows)


def cost_per_success(rows):
    successes = [r for r in rows if r["success"] == "True"]
    total_cost = sum(float(r["cost_usd"]) for r in rows)
    return total_cost / len(successes) if successes else float("inf")


METRICS = {
    "success_rate": success_rate,
    "cost_per_task": cost_per_task,
    "cost_per_success": cost_per_success,
}


def group_by_task(rows):
    """Group trial-rows by task_id -- tasks are the independent unit;
    the (usually 2) trials per task are correlated, so we resample whole
    tasks, not individual trial-rows, to avoid understating variance."""
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    return list(by_task.values())


def bootstrap_ci(rows, metric_fn, n_boot=N_BOOT):
    point = metric_fn(rows)
    tasks = group_by_task(rows)
    n_tasks = len(tasks)
    boots = []
    for _ in range(n_boot):
        sampled_tasks = [tasks[random.randrange(n_tasks)] for _ in range(n_tasks)]
        sample = [row for task_rows in sampled_tasks for row in task_rows]
        val = metric_fn(sample)
        if val != float("inf"):
            boots.append(val)
    boots.sort()
    if not boots:
        return point, float("inf"), float("inf")
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(int(0.975 * len(boots)), len(boots) - 1)]
    return point, lo, hi


def main():
    src = Path(__file__).parent / SOURCE
    rows = load_rows(src)

    by_group_router = {}
    for r in rows:
        key = (group_key(r["benchmark_name"]), r["router_name"])
        by_group_router.setdefault(key, []).append(r)

    out_path = Path(__file__).parent / "output" / "bootstrap_ci.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "benchmark_group", "metric", "point_estimate", "ci_low", "ci_high", "n_tasks"])
        for (group, router), group_rows in sorted(by_group_router.items()):
            n_tasks = len(group_by_task(group_rows))
            for metric_name, fn in METRICS.items():
                point, lo, hi = bootstrap_ci(group_rows, fn)
                w.writerow([router, group, metric_name, f"{point:.4f}", f"{lo:.4f}", f"{hi:.4f}", n_tasks])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
