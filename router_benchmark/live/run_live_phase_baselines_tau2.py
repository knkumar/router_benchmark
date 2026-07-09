"""Phase B Task 5: evaluate the 5 simple baseline routers on tau2-bench
(n_trials=1, matching this paper's existing tau2-bench convention), at
the Task 7-expanded n_tasks=25. AlwaysStrongestRouter is the single
biggest cost driver in this whole plan (every task runs the full
multi-turn session at claude-opus-4-8) -- run this script last, after
confirming remaining budget against the $70 combined ceiling."""

from __future__ import annotations

from router_benchmark.live.baseline_routers import build_all_baselines
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive


def main() -> None:
    run_live_phase(
        "baselines_tau2_v1",
        build_all_baselines(),
        [Tau2BenchLive(n_tasks=25)],
        seed=1234,
        n_trials=1,
    )


if __name__ == "__main__":
    main()
