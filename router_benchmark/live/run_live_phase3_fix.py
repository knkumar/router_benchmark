"""Phase 3 (fixed) live entry point: re-run of LiteLLM Router + Aurelio
Semantic Router against real tau2-bench, after the same two adapter
fixes described in run_live_phase1_fix.py. RouteLLM and vLLM Semantic
Router rows from the original phase3 are reused unchanged.

    python -m router_benchmark.live.run_live_phase3_fix
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive()]
    benchmarks = [Tau2BenchLive(n_tasks=8)]
    run_live_phase("phase3_fix", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
