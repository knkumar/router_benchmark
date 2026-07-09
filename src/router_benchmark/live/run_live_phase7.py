"""Phase 7 live entry point: WebArena (real self-hosted GitLab + Shopping
Docker sites, real Playwright browser, real evaluator) against all six
shortlisted live routers.

    python -m router_benchmark.live.run_live_phase7
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
from router_benchmark.live.webarena_live import WebArenaLive


def main() -> None:
    routers = [
        LiteLLMRouterLive(),
        AurelioSemanticRouterLive(),
        RouteLLMLive(),
        LLMRouterLive(),
        NVIDIABlueprintRouterLive(),
        VLLMSemanticRouterLive(),
    ]
    benchmarks = [WebArenaLive(n_tasks=8, sites=("gitlab", "shopping"))]
    run_live_phase("phase7", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
