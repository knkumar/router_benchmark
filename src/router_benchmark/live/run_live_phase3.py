"""Phase 3 live entry point: all 4 real routers (LiteLLM, Aurelio, RouteLLM,
vLLM Semantic Router) against real tau2-bench (retail domain).

Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, Docker running with the vLLM
Semantic Router service already started (see live/vllm_sr/README), and
router_benchmark/live/tau2env/tau2-bench set up via `uv sync`.

Small task/trial counts (n_tasks=8, n_trials=1) since each task is a real
multi-turn simulation costing ~30-45s wall-clock and a few cents.

    python -m router_benchmark.live.run_live_phase3
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(), VLLMSemanticRouterLive()]
    benchmarks = [Tau2BenchLive(n_tasks=8)]
    run_live_phase("phase3", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
