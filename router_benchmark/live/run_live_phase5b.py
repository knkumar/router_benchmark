"""Phase 5b live entry point: Terminal-Bench 2.0, all 6 shortlisted
routers against the curated 8-task easy subset (see
terminal_bench_live.py for the curation rationale). This is the first
full live phase for Terminal-Bench 2.0 -- previously only smoke-tested.

    python -m router_benchmark.live.run_live_phase5b
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.terminal_bench_live import TerminalBenchLive
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive


def main() -> None:
    routers = [
        LiteLLMRouterLive(),
        AurelioSemanticRouterLive(),
        RouteLLMLive(),
        LLMRouterLive(),
        NVIDIABlueprintRouterLive(),
        VLLMSemanticRouterLive(),
    ]
    benchmarks = [TerminalBenchLive(n_tasks=8)]
    run_live_phase("phase5b", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
