"""Phase 4b live entry point: the 4 remaining shortlisted routers
(Aurelio Semantic Router, vLLM Semantic Router, NVIDIA AI Blueprint LLM
Router, LLMRouter) against real SWE-bench Verified (same 2 real astropy
instances as Phase 4, zero-shot patch generation -- see
swebench_live.py module docstring for the zero-shot-baseline
limitation), completing SWE-bench coverage to all 6 shortlisted routers
alongside Phase 4's LiteLLM Router/RouteLLM results.

Expect several minutes of real Docker test-suite execution per
task-trial (image pull is one-time per instance since both routers'
runs share the same 2 astropy instances as Phase 4).

    python -m router_benchmark.live.run_live_phase4b
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.swebench_live import build_swebench_live
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive


def main() -> None:
    routers = [
        AurelioSemanticRouterLive(),
        VLLMSemanticRouterLive(),
        NVIDIABlueprintRouterLive(),
        LLMRouterLive(),
    ]
    benchmarks = [build_swebench_live(n_tasks=2)]
    run_live_phase("phase4b", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
