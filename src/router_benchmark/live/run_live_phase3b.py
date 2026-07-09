"""Phase 3b live entry point: the 2 remaining shortlisted routers
(NVIDIA AI Blueprint LLM Router, LLMRouter) against real tau2-bench
(retail domain), completing tau2-bench coverage to all 6 shortlisted
routers alongside Phase 3's LiteLLM Router/Aurelio/RouteLLM/vLLM Semantic
Router results.

Same task pool/seed/trial count as Phase 3 for direct comparability.

    python -m router_benchmark.live.run_live_phase3b
"""

from __future__ import annotations

from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive


def main() -> None:
    routers = [NVIDIABlueprintRouterLive(), LLMRouterLive()]
    benchmarks = [Tau2BenchLive(n_tasks=8)]
    run_live_phase("phase3b", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
