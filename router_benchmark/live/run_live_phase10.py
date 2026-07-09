"""Phase 10 live entry point: tau2-bench (real multi-turn retail-domain
simulation) against all six shortlisted live routers, at a larger sample
size than Phase 3's n=8 for improved statistical power, and extending
coverage to the two routers (LLMRouter, NVIDIA AI Blueprint LLM Router)
not previously run against tau2-bench.

Real per-task cost varies sharply by router (measured via a 3-task pilot,
see output/live/tau2_pilot/): LLMRouter ~$0.0045/task, RouteLLM ~$0.0057/
task, vLLM Semantic Router ~$0.213/task, Aurelio Semantic Router ~$0.262/
task, NVIDIA AI Blueprint LLM Router ~$0.437/task, LiteLLM Router ~$0.520/
task -- at n_tasks=25 this totals an estimated ~$36 real spend.

    python -m router_benchmark.live.run_live_phase10
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive
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
    benchmarks = [Tau2BenchLive(n_tasks=100)]
    run_live_phase("phase10", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
