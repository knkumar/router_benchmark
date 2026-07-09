"""Phase 8 live entry point: SWE-bench Verified (real Docker container
grading) against all six shortlisted live routers, at a larger sample size
than Phase 4's n=2 for improved statistical power.

Uses SWEBenchLive(n_tasks=20) directly (not build_swebench_live's fixed
2-astropy-instance shortcut): pulls the first 20 of the real 194
"<15 min fix" real SWE-bench Verified instances in dataset order, spanning
several real repos (django, sympy, sphinx, matplotlib, ...), not just
astropy. Real $ cost is negligible (zero-shot patch generation only, ~$0.001/
task); the real cost here is wall-clock (each task is a real Docker
container build + real project test suite run).

    python -m router_benchmark.live.run_live_phase8
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.swebench_live import SWEBenchLive
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
    benchmarks = [SWEBenchLive(n_tasks=20)]
    run_live_phase("phase8", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
