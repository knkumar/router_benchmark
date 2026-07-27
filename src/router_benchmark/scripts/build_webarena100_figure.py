"""Rebuild the WebArena-only Pareto figure for paper1.tex from phase9
(n=100, 2026-07-08), restricted to this paper's 4-router scope. Replaces
the stale phase7c_v2 (n=16) figure that paper1_webarena_pareto_frontier.png
was generated from before the WebArena scale-up.

    python -m router_benchmark.build_webarena100_figure
"""

from __future__ import annotations

from pathlib import Path

from router_benchmark.scripts._paths import repository_root

REPO_ROOT = repository_root()

import pandas as pd

from router_benchmark.metrics import (
    compute_pareto_frontier,
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots

PAPER_ROUTERS = {
    "RouteLLM (live)",
    "LiteLLM Router (live)",
    "vLLM Semantic Router (live)",
    "Aurelio Semantic Router (live)",
}

LIVE_ROOT = REPO_ROOT / "output" / "live"
OUT_DIR = LIVE_ROOT / "phase9_v1"


def main() -> None:
    df = pd.read_csv(LIVE_ROOT / "phase9" / "results.csv")
    df = df[df["router_name"].isin(PAPER_ROUTERS)]
    assert set(df["router_name"]) == PAPER_ROUTERS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "results.csv", index=False)

    per_bm_df = compute_router_benchmark_metrics(df)
    per_bm_df.to_csv(OUT_DIR / "metrics_per_benchmark.csv", index=False)

    overall_df = compute_router_overall_metrics(df)
    overall_df.to_csv(OUT_DIR / "metrics_overall.csv", index=False)

    frontier_df = compute_pareto_frontier(overall_df)
    frontier_df.to_csv(OUT_DIR / "pareto_frontier.csv", index=False)
    print(frontier_df[["router_name", "mean_success_rate", "mean_cost_per_task_usd", "is_pareto_optimal"]].to_string(index=False))

    plots_dir = OUT_DIR / "plots"
    generate_all_plots(df, per_bm_df, overall_df, plots_dir, label="WebArena, n=100")
    print(f"\nOutput written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
