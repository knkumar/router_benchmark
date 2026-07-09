#!/usr/bin/env python3
"""Failure taxonomy, mined from existing per-task result data -- no
re-run needed. This harness's results.csv schema has no dedicated
`error_type` column, but each adapter's own exception-handling pattern
(confirmed by source inspection: router_benchmark/live/live_benchmarks.py
BFCLLive.score's except block, router_benchmark/live/tau2_live.py
Tau2BenchLive.score's non-zero-returncode and JSON-parse-failure paths)
reports `cost_usd == 0.0` specifically when the real API/subprocess call
failed before being billed -- a real infra/API-level failure, distinct
from a graded wrong answer (which still costs money). We categorize:

  - "infra_or_api_failure": success == False and cost_usd == 0.0
  - "graded_incorrect":     success == False and cost_usd  > 0.0
  - "success":              success == True

This is coarser than the full taxonomy the review asked for (router
error / API error / tool-call format error / browser navigation error /
grader failure / timeout / fallback are not separately distinguishable
without adding new instrumentation to each adapter -- a Phase C item if
finer granularity is wanted), but it is a real, non-fabricated split
derivable from data already collected, which is why it ships now instead
of waiting on a new paid run.
"""
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


def categorize(row):
    if row["success"] == "True":
        return "success"
    cost = float(row["cost_usd"]) if row["cost_usd"] not in ("", "nan") else 0.0
    if cost == 0.0:
        return "infra_or_api_failure"
    return "graded_incorrect"


def main():
    src = Path(__file__).parent / "../output/live/paper1_live_v2/results.csv"
    with open(src) as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        group = group_key(r["benchmark_name"])
        counts[group][categorize(r)] += 1

    out_path = Path(__file__).parent / "output" / "failure_taxonomy.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark_group", "success", "graded_incorrect", "infra_or_api_failure", "n_total"])
        for group, cat_counts in sorted(counts.items()):
            n_total = sum(cat_counts.values())
            w.writerow([
                group,
                cat_counts.get("success", 0),
                cat_counts.get("graded_incorrect", 0),
                cat_counts.get("infra_or_api_failure", 0),
                n_total,
            ])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
