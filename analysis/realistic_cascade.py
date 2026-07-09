#!/usr/bin/env python3
"""Realistic (non-oracle) cheap-first cascade: escalates cheap-small ->
mid-general -> strong-frontier only on a signal a real deployed system
could observe WITHOUT knowing whether the task was ultimately solved
correctly.

Signal-availability audit (done before writing this): `tool_call_correct`
in this harness (router_benchmark/live/live_benchmarks.py BFCLLive.score,
router_benchmark/live/tau2_live.py Tau2BenchLive.score) is graded against
ground truth (BFCL: `correct = self._grade(tool_calls, ground_truth, ...)`
and `tool_call_correct` is set to that same graded value; tau2:
`tool_call_correct = bool(reward >= 0.5)`, also graded) -- so it CANNOT be
used as a pre-answer escalation trigger without smuggling oracle
information back in. This rules out Task 17's Step 2a path.

The one real, pre-answer, cross-benchmark signal actually present in the
existing per-task data is `cost_usd == 0.0`: every adapter's exception
handler (BFCLLive's API-error except block, Tau2BenchLive's non-zero
subprocess returncode / JSON-parse-failure paths) reports zero cost
specifically because the request failed or errored *before* the call
would have been billed -- a real infrastructure/API-level failure
signal, not a graded-correctness one. WebArena is a partial exception:
its timeout path still reports whatever real cost had already accrued
(router_benchmark/live/webarena_live.py's `timed_out` branch calls
`_read_trace_cost` unconditionally), so this heuristic has no
comparable trigger there -- documented as a known limitation below,
not silently papered over.
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

    out_path = Path(__file__).parent / "output" / "realistic_cascade.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark_name", "task_id", "realistic_cascade_success", "realistic_cascade_cost", "tiers_tried"])
        for (bench, task_id), tiers in sorted(by_task.items()):
            cost = 0.0
            tried = []
            final_success = False
            for tier in TIER_ORDER:
                if tier not in tiers:
                    continue
                row = tiers[tier]
                tried.append(tier)
                tier_cost = float(row["cost_usd"])
                cost += tier_cost
                infra_failure = (tier_cost == 0.0)
                if not infra_failure:
                    # No oracle-free signal to escalate further on (this
                    # benchmark's tiers are not WebArena-only exceptions to
                    # that rule) -- accept this tier's real outcome as final.
                    final_success = row["success"] == "True"
                    break
                # else: tier_cost == 0.0 (real infra/API failure signal,
                # not graded correctness) -> escalate to the next tier.
            else:
                # Exhausted all tiers with zero cost every time (real
                # infra failure at every tier) -- report the last tier's
                # real (failed) outcome.
                final_success = tiers[tried[-1]]["success"] == "True" if tried else False

            w.writerow([bench, task_id, final_success, f"{cost:.6f}", "+".join(tried)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
