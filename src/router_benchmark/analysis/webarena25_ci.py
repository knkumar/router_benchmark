#!/usr/bin/env python3
"""Bootstrap 95% CIs for the Phase B Task 7 WebArena expansion (n=25 tasks,
phase7_expanded/results.csv), same method as bootstrap_ci.py Task 1."""
import csv
import random
from pathlib import Path

random.seed(20260705)
N_BOOT = 10000

SRC = Path(__file__).parent / "../output/live/phase7_expanded/results.csv"


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


def bootstrap_ci(rows, metric_fn, n_boot=N_BOOT):
    point = metric_fn(rows)
    n = len(rows)
    boots = sorted(metric_fn([rows[random.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return point, lo, hi


def main():
    with open(SRC) as f:
        rows = list(csv.DictReader(f))
    by_router = {}
    for r in rows:
        by_router.setdefault(r["router_name"], []).append(r)

    out_path = Path(__file__).parent / "output" / "webarena25_ci.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "metric", "point_estimate", "ci_low", "ci_high", "n_tasks"])
        for router, group_rows in sorted(by_router.items()):
            for metric_name, fn in METRICS.items():
                point, lo, hi = bootstrap_ci(group_rows, fn)
                w.writerow([router, metric_name, f"{point:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(group_rows)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

