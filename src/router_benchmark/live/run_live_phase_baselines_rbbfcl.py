"""Phase B Task 5: evaluate the 5 simple baseline routers on RouterBench +
BFCL (n_trials=2, matching this paper's existing RouterBench/BFCL trial
convention)."""

from __future__ import annotations

from router_benchmark.live.baseline_routers import build_all_baselines
from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.run_common import run_live_phase


def main() -> None:
    run_live_phase(
        "baselines_rbbfcl_v1",
        build_all_baselines(),
        build_live_benchmarks(routerbench_n=60, bfcl_n=30),
        seed=1234,
        n_trials=2,
    )


if __name__ == "__main__":
    main()
