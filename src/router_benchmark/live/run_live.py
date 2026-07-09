"""Phase 1 live entry point: LiteLLM Router + Aurelio Semantic Router
(real pip-installable routers) against RouterBench + BFCL v4 (real data).

    python -m router_benchmark.live.run_live

Requires OPENAI_API_KEY and ANTHROPIC_API_KEY in the environment. Writes to
router_benchmark/output/live/phase1/ (manifest.json, traces.jsonl,
results.csv, metrics_*.csv, plots/*.png) -- see run_common.py.
"""

from __future__ import annotations

from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.live_routers import build_live_routers
from router_benchmark.live.run_common import run_live_phase


def main() -> None:
    routers = build_live_routers()
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase("phase1", routers, benchmarks, seed=1234, n_trials=2)


if __name__ == "__main__":
    main()
