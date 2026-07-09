"""Phase B Task 5: evaluate the 5 simple baseline routers on WebArena
(n_trials=1, matching this paper's existing WebArena convention), at the
Task 7-expanded n_tasks=25."""

from __future__ import annotations

from router_benchmark.live.baseline_routers import build_all_baselines
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.webarena_live import WebArenaLive


def main() -> None:
    run_live_phase(
        "baselines_webarena_v1",
        build_all_baselines(),
        [WebArenaLive(n_tasks=25, sites=("gitlab", "shopping"))],
        seed=1234,
        n_trials=1,
    )


if __name__ == "__main__":
    main()
