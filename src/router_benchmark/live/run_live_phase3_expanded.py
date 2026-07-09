"""Phase B Task 7: tau2-bench at n_tasks=25 (up from the original n=8),
against this paper's 4 real routers only."""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(), VLLMSemanticRouterLive()]
    benchmarks = [Tau2BenchLive(n_tasks=25)]
    run_live_phase("phase3_expanded", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
