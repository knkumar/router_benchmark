#!/usr/bin/env python3
"""Operating metrics for the idealized cheap->mid->strong cascade (existing-data
only), computed from a locked canonical bundle's exhaustive candidate matrix.

Reviewer point: the idealized cascade is reported by success (and cost) alone,
which is not comparable to a one-call router because the cascade may execute
several models. This script adds, per benchmark, the average number of model
calls, the mean summed latency, and success-per-dollar, so the cascade's success
can be read against what it spends to get there. The idealized cascade escalates
on the real grader's success verdict (an upper bound that a deployed system
cannot obtain before generation); the realistic cascade escalates only on a
pre-answer infra-failure signal (billed cost == 0) and is reported alongside.

This writes only the analysis CSV. The paper LaTeX fragment is rendered from that
CSV by scripts/generate_paper1_canonical_tables.py (_write_cascade_operating), so
the canonical driver remains the single source of the paper tables.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

TIER_ORDER = ["cheap-small", "mid-general", "strong-frontier"]
SHORT = {
    "RouterBench (live)": "RouterBench",
    "BFCL v4 (live)": "BFCL",
    "tau2-bench (live)": "tau2-bench",
    "WebArena (live)": "WebArena",
}
DISPLAY = ["RouterBench", "BFCL", "tau2-bench", "WebArena"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument(
        "--csv-output",
        type=Path,
        default=Path(__file__).parent / "output" / "paper1_canonical" / "cascade_operating_metrics.csv",
    )
    args = ap.parse_args()

    reps: dict[tuple[str, str, str], list[tuple[bool, float, float]]] = defaultdict(list)
    for r in _read(args.bundle / "candidate_outcomes.csv"):
        reps[(r["benchmark_id"], r["task_id"], r["candidate_id"])].append(
            (
                r["success"].lower() == "true",
                float(r["model_api_cost_usd"]),
                float(r["generation_latency_ms"]) / 1000.0,
            )
        )
    benches = sorted({b for (b, _, _) in reps})
    tasks_by_bench: dict[str, set[str]] = defaultdict(set)
    for (b, t, _) in reps:
        tasks_by_bench[b].add(t)

    stats: dict[str, tuple[float, float, float, float, float]] = {}
    for b in benches:
        succ = calls = lat = cost = 0.0
        n = 0
        for t in tasks_by_bench[b]:
            n_reps = max(len(reps[(b, t, x)]) for x in TIER_ORDER)
            for rep in range(n_reps):
                c_calls = c_cost = c_lat = 0.0
                solved = False
                for x in TIER_ORDER:
                    rr = reps[(b, t, x)]
                    if rep >= len(rr):
                        continue
                    ok, co, la = rr[rep]
                    c_calls += 1
                    c_cost += co
                    c_lat += la
                    if ok:  # idealized: escalate until the grader reports success
                        solved = True
                        break
                succ += 1.0 if solved else 0.0
                calls += c_calls
                lat += c_lat
                cost += c_cost
                n += 1
        s = succ / n
        c = cost / n
        stats[SHORT[b]] = (s, calls / n, lat / n, c, (s / c if c > 0 else float("nan")))

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "success", "avg_calls", "avg_latency_s", "cost_per_task_usd", "success_per_usd"])
        # High precision so the driver's display rounding matches the true value.
        for name in DISPLAY:
            s, ca, la, co, spd = stats[name]
            w.writerow([name, f"{s:.10f}", f"{ca:.10f}", f"{la:.6f}", f"{co:.10f}", f"{spd:.6f}"])

    print(f"wrote {args.csv_output}")
    for name in DISPLAY:
        print(name, stats[name])


if __name__ == "__main__":
    main()
