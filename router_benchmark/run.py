"""Entry point: run the full evaluation, write results/metrics to disk, and
generate plots. See README.md for details.

    python -m router_benchmark.run
"""

from __future__ import annotations

from pathlib import Path

from router_benchmark.benchmarks import build_all_benchmarks
from router_benchmark.harness import EvaluationHarness
from router_benchmark.metrics import (
    compute_pareto_frontier,
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots
from router_benchmark.routers import build_all_routers

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    routers = build_all_routers()
    benchmarks = build_all_benchmarks()

    print(f"Evaluating {len(routers)} routers x {len(benchmarks)} benchmarks ...")
    harness = EvaluationHarness(seed=1234, n_trials=3)
    results_df = harness.evaluate(routers, benchmarks)
    print(f"  -> {len(results_df):,} task-trial rows")

    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    per_bm_df = compute_router_benchmark_metrics(results_df)
    per_bm_df.to_csv(OUTPUT_DIR / "metrics_per_benchmark.csv", index=False)

    overall_df = compute_router_overall_metrics(results_df)
    overall_df.to_csv(OUTPUT_DIR / "metrics_overall.csv", index=False)

    frontier_df = compute_pareto_frontier(overall_df)
    frontier_df.to_csv(OUTPUT_DIR / "pareto_frontier.csv", index=False)

    print("\n=== Overall router ranking (mean success rate) ===")
    cols = ["router_name", "mean_success_rate", "mean_cost_per_task_usd", "mean_latency_p50_ms", "is_pareto_optimal"]
    print(overall_df[cols].to_string(index=False))

    print(f"\nPareto-optimal routers: {', '.join(frontier_df['router_name'])}")

    plot_paths = generate_all_plots(results_df, per_bm_df, overall_df, PLOTS_DIR)
    print("\nWrote plots:")
    for p in plot_paths:
        print(f"  {p}")

    print(f"\nAll output written under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
