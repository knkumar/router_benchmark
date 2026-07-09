"""Phase 7b live entry point: WebArena expanded-sample follow-up to Phase 7
(16 tasks/router instead of 8, for more statistical power on the directional
ranking Phase 7 produced). Same routers, same sites, same seed (the task
pool is a fixed-order prefix slice, so these 16 tasks include the original
8 as their first 8 -- Phase 7's rows are not reused here, this is an
independent rerun at the larger sample size).

    python -m router_benchmark.live.run_live_phase7b
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
    benchmarks = [WebArenaLive(n_tasks=16, sites=("gitlab", "shopping"))]
    run_live_phase("phase7b", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
