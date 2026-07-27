"""Build the current Paper 1 live-evidence package.

This is the canonical post-cache-audit Paper 1 package. It starts from
``output/live/paper1_live_v2/results.csv`` and replaces only the superseded
vLLM Semantic Router tau2-bench slice with the fresh cache-bypassed run in
``output/live/tau2_vllm_fresh_v1/results.csv``.

The output directory is ``output/live/paper1_live_v3/``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from router_benchmark.scripts._paths import repository_root

ROOT = repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.metrics import (
    compute_pareto_frontier,
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots

LIVE_ROOT = ROOT / "output" / "live"
OUT_DIR = LIVE_ROOT / "paper1_live_v3"

VLLM = "vLLM Semantic Router (live)"
TAU2 = "tau2-bench (live)"


def _is_vllm_tau2(df: pd.DataFrame) -> pd.Series:
    return (df["router_name"] == VLLM) & (df["benchmark_name"] == TAU2)


def main() -> None:
    base = pd.read_csv(LIVE_ROOT / "paper1_live_v2" / "results.csv")
    fresh = pd.read_csv(LIVE_ROOT / "tau2_vllm_fresh_v1" / "results.csv")
    fresh = fresh[_is_vllm_tau2(fresh)].copy()

    if len(fresh) != 100:
        raise SystemExit(f"Expected 100 fresh vLLM tau2 rows, found {len(fresh)}")

    old = base[_is_vllm_tau2(base)]
    if len(old) != 100:
        raise SystemExit(f"Expected 100 old vLLM tau2 rows, found {len(old)}")

    combined = pd.concat([base[~_is_vllm_tau2(base)], fresh], ignore_index=True)
    combined = combined.sort_values(["benchmark_name", "router_name", "trial", "task_id"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DIR / "results.csv", index=False)

    per_bm = compute_router_benchmark_metrics(combined)
    per_bm.to_csv(OUT_DIR / "metrics_per_benchmark.csv", index=False)

    overall = compute_router_overall_metrics(combined)
    overall.to_csv(OUT_DIR / "metrics_overall.csv", index=False)

    frontier = compute_pareto_frontier(overall)
    frontier.to_csv(OUT_DIR / "pareto_frontier.csv", index=False)

    generate_all_plots(combined, per_bm, overall, OUT_DIR / "plots", label="live evaluation (Paper 1 v3)")

    summary = combined[_is_vllm_tau2(combined)]["success"].agg(["sum", "count", "mean"])
    print(f"Wrote {OUT_DIR}")
    print(f"vLLM tau2 fresh rows: {int(summary['sum'])}/{int(summary['count'])} = {summary['mean']:.3f}")
    print(f"vLLM tau2 cost/task: {combined[_is_vllm_tau2(combined)]['cost_usd'].mean():.4f}")


if __name__ == "__main__":
    main()
