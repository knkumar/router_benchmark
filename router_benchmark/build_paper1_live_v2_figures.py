"""Build the two benchmark-group-specific output dirs (RouterBench+BFCL,
WebArena) needed for paper1.tex's two separate Pareto-frontier figures,
using the fixed LiteLLM Router + Aurelio Semantic Router adapters and the
unchanged RouteLLM + vLLM Semantic Router rows -- mirrors how the original
combined_rb_bfcl/ and phase7c/ each independently produced
plots/pareto_frontier.png over only their own benchmark(s).

    python -m router_benchmark.build_paper1_live_v2_figures
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from router_benchmark.metrics import (
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots

FIXED_ROUTERS = {"LiteLLM Router (live)", "Aurelio Semantic Router (live)"}
UNCHANGED_ROUTERS = {"RouteLLM (live)", "vLLM Semantic Router (live)"}

LIVE_ROOT = Path(__file__).parent / "output" / "live"

GROUPS = {
    "combined_rb_bfcl_v2": ("phase1_fix", ["phase2", "phase2b"], "RouterBench + BFCL v4, n=90 tasks / 180 trials"),
    "phase7c_v2": ("phase7c_fix", ["phase7c"], "WebArena pilot, n=16"),
}


def main() -> None:
    for out_name, (fixed_phase, unchanged_phases, label) in GROUPS.items():
        out_dir = LIVE_ROOT / out_name
        out_dir.mkdir(parents=True, exist_ok=True)

        fixed_df = pd.read_csv(LIVE_ROOT / fixed_phase / "results.csv")
        fixed_df = fixed_df[fixed_df["router_name"].isin(FIXED_ROUTERS)]
        frames = [fixed_df]
        for unchanged_phase in unchanged_phases:
            unchanged_df = pd.read_csv(LIVE_ROOT / unchanged_phase / "results.csv")
            unchanged_df = unchanged_df[unchanged_df["router_name"].isin(UNCHANGED_ROUTERS)]
            frames.append(unchanged_df)

        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out_dir / "results.csv", index=False)

        per_bm_df = compute_router_benchmark_metrics(combined)
        per_bm_df.to_csv(out_dir / "metrics_per_benchmark.csv", index=False)

        overall_df = compute_router_overall_metrics(combined)
        overall_df.to_csv(out_dir / "metrics_overall.csv", index=False)

        plots_dir = out_dir / "plots"
        generate_all_plots(combined, per_bm_df, overall_df, plots_dir, label=label)
        print(f"{out_name}: {len(combined)} rows, benchmarks={sorted(combined['benchmark_name'].unique())}")
        print(overall_df[["router_name", "mean_success_rate", "mean_cost_per_task_usd", "is_pareto_optimal"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
