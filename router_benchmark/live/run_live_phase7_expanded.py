"""Phase B Task 7: WebArena at n_tasks=25 (up from the original n=16),
against this paper's 4 real routers only (not the 6-router shortlist from
earlier phase9/phase10 scripts, which include routers out of this paper's
scope)."""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
from router_benchmark.live.webarena_live import WebArenaLive


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(), VLLMSemanticRouterLive()]
    benchmarks = [WebArenaLive(n_tasks=25, sites=("gitlab", "shopping"))]
    run_live_phase("phase7_expanded", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
