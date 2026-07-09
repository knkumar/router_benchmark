"""Phase 11 live entry point: a second trial (trial index 1) of tau2-bench
for the two routers whose Phase 10 tau2 runs needed re-running -- vLLM
Semantic Router and NVIDIA AI Blueprint LLM Router -- to add variance-aware
coverage on top of Phase 10's single trial (trial index 0, all six routers).

Real per-task cost for these two is the most expensive on tau2 (vLLM
~$0.213/task, NVIDIA ~$0.437/task); one trial of n=100 each totals an
estimated ~$65 real spend, so run deliberately.

    python -m router_benchmark.live.run_live_phase11

Reproducing Phase 11
--------------------
The original Phase 11 results.csv held only trial-1 rows because trial 0 for
these two routers had already been computed in Phase 10. The original run got
that skip via a hardcoded, global phase10 look-up baked into the harness --
which silently contaminated every other run (see run_common.run_live_phase /
EvaluationHarness.evaluate for the corrected, opt-in mechanism).

Default (resume) path here: seed this phase's own results_incremental.csv
with Phase 10's trial-0 rows for these two routers, then run with
resume=True. Trial 0 is skipped as already-done (no re-spend) and trial 1 is
computed and appended, so phase11/results.csv ends up with both trials for
the two routers; the paper's Phase 11 rows are the trial==1 subset. Because
seeds are derived per (router, benchmark, trial), those trial-1 rows match
the original regardless of how trial 0 was obtained. Pass --fresh to instead
recompute both trials from scratch without touching Phase 10.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.run_common import LIVE_OUTPUT_ROOT, run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive

PHASE11_ROUTERS = ("vLLM Semantic Router (live)", "NVIDIA AI Blueprint LLM Router (live)")


def seed_phase11_from_phase10() -> int:
    """Copy Phase 10's trial-0 rows for the two Phase 11 routers into Phase
    11's own results_incremental.csv, so a resume run skips them instead of
    re-spending on trial 0. Returns the number of rows seeded."""
    p10 = LIVE_OUTPUT_ROOT / "phase10" / "results_incremental.csv"
    if not p10.exists():
        return 0
    with open(p10, newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            r for r in reader
            if r["router_name"] in PHASE11_ROUTERS
            and r["benchmark_name"] == "tau2-bench (live)"
            and str(r.get("trial", "0")) == "0"
        ]
        fieldnames = reader.fieldnames
    if not rows:
        return 0
    out_dir = LIVE_OUTPUT_ROOT / "phase11"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_incremental.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    fresh = "--fresh" in sys.argv
    routers = [VLLMSemanticRouterLive(), NVIDIABlueprintRouterLive()]
    benchmarks = [Tau2BenchLive(n_tasks=100)]
    if fresh:
        # Recompute both trials from scratch (400 rows; ~$130 spend).
        run_live_phase("phase11", routers, benchmarks, seed=1234, n_trials=2, resume=False)
    else:
        seeded = seed_phase11_from_phase10()
        print(f"Seeded {seeded} Phase 10 trial-0 rows; computing trial 1 only via resume.")
        run_live_phase("phase11", routers, benchmarks, seed=1234, n_trials=2, resume=True)


if __name__ == "__main__":
    main()
