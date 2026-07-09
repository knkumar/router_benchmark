"""Phase B Task 1: exhaustive candidate-tier sweep -- runs each of the 3
candidate tiers (cheap-small, mid-general, strong-frontier) on every real
task across all 4 benchmarks, once, via ForcedTierRouter. This is the
shared ground-truth table for the oracle upper bound, pessimal bound,
regret-to-oracle, normalized-cost, and realistic-cascade analyses.
Sized to match Task 7's expanded tau2-bench/WebArena sample counts
(n_tasks=25 each) so this data is directly comparable to the expanded
4-router runs, not the original n=8/n=16 pilot scale.
"""

from __future__ import annotations

from router_benchmark.live.candidate_sweep import ForcedTierRouter
from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.webarena_live import WebArenaLive

TIERS = ["cheap-small", "mid-general", "strong-frontier"]


def main() -> None:
    routers = [ForcedTierRouter(t) for t in TIERS]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30) + [
        Tau2BenchLive(n_tasks=25),
        WebArenaLive(n_tasks=25, sites=("gitlab", "shopping")),
    ]
    run_live_phase(
        "sweep_v1",
        routers,
        benchmarks,
        seed=1234,
        n_trials=1,
        extra_manifest={"purpose": "exhaustive candidate-tier sweep for oracle/regret/cascade/normalized-cost baselines"},
    )


if __name__ == "__main__":
    main()
