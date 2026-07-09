"""Phase 6 live entry point: LLMRouter (real KNN router trained on real
bundled example data, real OpenAI-embedding substitution for the platform-
blocked Longformer encoder -- see llmrouter_live.py module docstring)
against RouterBench + BFCL v4.

    python -m router_benchmark.live.run_live_phase6
"""

from __future__ import annotations

from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.run_common import run_live_phase


def main() -> None:
    routers = [LLMRouterLive()]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase("phase6", routers, benchmarks, seed=1234, n_trials=2)


if __name__ == "__main__":
    main()
