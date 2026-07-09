"""Phase 2 (partial) live entry point: RouteLLM (real sw_ranking router,
real Chatbot Arena data) added against the same real RouterBench + BFCL v4
benchmarks used in Phase 1.

    python -m router_benchmark.live.run_live_phase2

Requires OPENAI_API_KEY and ANTHROPIC_API_KEY. Writes to
router_benchmark/output/live/phase2/.

NOTE: this covers only RouteLLM so far. vLLM Semantic Router, LLMRouter,
and the NVIDIA AI Blueprint Router each need substantially heavier
integration (a Go/Envoy service, a full training-data/config pipeline, and
a notebook/NIM-microservices deployment respectively) and tau2-bench needs
a separate Python >=3.12 environment -- see router_benchmark/live/README.md
for the concrete plan once scope is confirmed.
"""

from __future__ import annotations

from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase


def main() -> None:
    routers = [RouteLLMLive()]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase("phase2", routers, benchmarks, seed=1234, n_trials=2)


if __name__ == "__main__":
    main()
