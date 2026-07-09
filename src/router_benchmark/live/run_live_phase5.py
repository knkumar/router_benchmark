"""Phase 5 live entry point: NVIDIA AI Blueprint LLM Router (real intent-
classification routing logic from the blueprint, real live classifier
call) against RouterBench + BFCL v4.

    python -m router_benchmark.live.run_live_phase5
"""

from __future__ import annotations

from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.run_common import run_live_phase


def main() -> None:
    routers = [NVIDIABlueprintRouterLive()]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase("phase5", routers, benchmarks, seed=1234, n_trials=2)


if __name__ == "__main__":
    main()
