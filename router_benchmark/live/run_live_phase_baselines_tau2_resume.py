"""Resume of baselines_tau2_v1 after a host-memory-pressure crash: the
Always-Cheapest and Always-Strongest baselines already completed all 25
tau2-bench tasks (see results_incremental.PRECRASH_partial.csv) before an
unrelated host process exhausted memory and killed a subprocess call.
Only the remaining 3 baselines are re-run here to avoid re-spending on
Always-Strongest (the single biggest cost driver in this phase)."""

from __future__ import annotations

from router_benchmark.live.baseline_routers import (
    PromptLengthHeuristicRouter,
    RandomRouter,
    ToolRequiredHeuristicRouter,
)
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive


def main() -> None:
    run_live_phase(
        "baselines_tau2_v1_resume",
        [RandomRouter(), PromptLengthHeuristicRouter(), ToolRequiredHeuristicRouter()],
        [Tau2BenchLive(n_tasks=25)],
        seed=1234,
        n_trials=1,
    )


if __name__ == "__main__":
    main()
