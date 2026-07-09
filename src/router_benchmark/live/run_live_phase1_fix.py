"""Phase 1 (fixed) live entry point: re-run of Phase 1 (LiteLLM Router +
Aurelio Semantic Router against RouterBench + BFCL v4) after fixing two
adapter bugs documented in live_routers.py and the
router_bench_adapter_bugs memory:

  - LiteLLMRouterLive previously never called the real litellm.Router it
    constructed; it now calls the real async cost-based-routing path.
  - AurelioSemanticRouterLive previously defaulted silently to
    "mid-general" (with a fabricated confidence=0.7) whenever
    semantic-router found no match above threshold; it now uses the
    package's real aggregation="max" option and logs genuine fallbacks
    honestly.

RouteLLM and vLLM Semantic Router are not re-run here: no bug was found
in their adapters. Same task/trial parameters as the original Phase 1
(seed=1234, n_trials=2, routerbench_n=60, bfcl_n=30) for an apples-to-
apples comparison against the unchanged phase2 (RouteLLM)/phase2b (vLLM)
rows.

    python -m router_benchmark.live.run_live_phase1_fix
"""

from __future__ import annotations

from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.run_common import run_live_phase


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive()]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase("phase1_fix", routers, benchmarks, seed=1234, n_trials=2)


if __name__ == "__main__":
    main()
