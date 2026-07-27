#!/usr/bin/env python3
"""Oracle upper bound, pessimal lower bound, and idealized cheap-first
cascade, computed entirely from the Task-1 exhaustive sweep -- no new API
calls. Oracle = best real outcome across all 3 tiers per task (success
first, then lowest cost as tiebreak). Pessimal = worst real outcome
(priciest failure if any tier fails, else cheapest success -- a
calibrated-to-be-bad baseline, the mirror image of oracle). Cascade =
try cheap-small first; if the real grader marked it a failure, escalate
to mid-general; if that also failed, escalate to strong-frontier. The
cascade is an idealized upper bound (it uses ground-truth success as its
escalation signal, which a real deployed cascade would not have access to
at decision time) -- documented explicitly as a ceiling on cascade
performance, not a deployable policy. See realistic_cascade.py for a
deployable variant using only pre-answer signals.
"""
import csv
from collections import defaultdict
from pathlib import Path

TIER_ORDER = ["cheap-small", "mid-general", "strong-frontier"]


def main():
    src = Path(__file__).parent / "../data/live/sweep_v1/sweep_results.csv"
    with open(src) as f:
        rows = list(csv.DictReader(f))

    by_task = defaultdict(dict)
    for r in rows:
        by_task[(r["benchmark_name"], r["task_id"])][r["candidate_tier"]] = r

    out_path = Path(__file__).parent / "output" / "oracle_and_cascade.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark_name", "task_id",
            "oracle_success", "oracle_cost", "oracle_tier",
            "pessimal_success", "pessimal_cost", "pessimal_tier",
            "cascade_success", "cascade_cost", "cascade_tiers_tried",
        ])
        for (bench, task_id), tiers in sorted(by_task.items()):
            # Oracle: any tier succeeds -> success; among successes, cheapest wins; else cheapest overall.
            successes = {t: row for t, row in tiers.items() if row["success"] == "True"}
            pool = successes if successes else tiers
            oracle_tier = min(pool, key=lambda t: float(pool[t]["cost_usd"]))
            oracle_row = tiers[oracle_tier]

            # Pessimal: priciest failure if any tier fails, else cheapest success.
            failures = {t: row for t, row in tiers.items() if row["success"] == "False"}
            if failures:
                pessimal_tier = max(failures, key=lambda t: float(failures[t]["cost_usd"]))
            else:
                pessimal_tier = min(tiers, key=lambda t: float(tiers[t]["cost_usd"]))
            pessimal_row = tiers[pessimal_tier]

            # Cheap-first idealized cascade: escalate through TIER_ORDER until success or exhausted.
            cascade_cost = 0.0
            cascade_success = False
            tried = []
            for tier in TIER_ORDER:
                if tier not in tiers:
                    continue
                tried.append(tier)
                cascade_cost += float(tiers[tier]["cost_usd"])
                if tiers[tier]["success"] == "True":
                    cascade_success = True
                    break

            w.writerow([
                bench, task_id,
                oracle_row["success"], oracle_row["cost_usd"], oracle_tier,
                pessimal_row["success"], pessimal_row["cost_usd"], pessimal_tier,
                cascade_success, f"{cascade_cost:.6f}", "+".join(tried),
            ])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
