"""Phase 7c (fixed) live entry point: re-run of LiteLLM Router + Aurelio
Semantic Router against real WebArena (self-hosted GitLab + Magento),
after the same two adapter fixes described in run_live_phase1_fix.py.
RouteLLM and vLLM Semantic Router rows from the original phase7c are
reused unchanged.

    python -m router_benchmark.live.run_live_phase7c_fix
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.webarena_live import WebArenaLive


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive()]
    benchmarks = [WebArenaLive(n_tasks=16, sites=("gitlab", "shopping"))]
    run_live_phase("phase7c_fix", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
