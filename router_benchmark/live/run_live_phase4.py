"""Phase 4 live entry point: 2 real routers (LiteLLM Router, RouteLLM)
against real SWE-bench Verified (2 real astropy instances, zero-shot
patch generation -- see swebench_live.py module docstring for the
zero-shot limitation).

Expect several minutes of real Docker test-suite execution per task-trial.

    python -m router_benchmark.live.run_live_phase4
"""

from __future__ import annotations

from router_benchmark.live.live_routers import LiteLLMRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.swebench_live import build_swebench_live


def main() -> None:
    routers = [LiteLLMRouterLive(), RouteLLMLive()]
    benchmarks = [build_swebench_live(n_tasks=2)]
    run_live_phase("phase4", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
