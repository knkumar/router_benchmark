"""Shared live-run driver: reproducibility plumbing used by every phase
script (run_live.py = Phase 1, run_live_phase2.py = Phase 2, ...).

For each phase, writes to router_benchmark/output/live/<phase>/:
    manifest.json     - full run config: models, pricing, seeds, sample
                         sizes, package versions, routers/benchmarks used
    traces.jsonl       - every real LLM API call: full request + full
                         response + cost/latency/timestamp (see
                         trace_logger.py)
    results.csv        - one row per (router, benchmark, task, trial),
                         same schema as the simulated harness output
    metrics_per_benchmark.csv / metrics_overall.csv / pareto_frontier.csv
    plots/*.png
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from router_benchmark.harness import EvaluationHarness
from router_benchmark.live.llm_client import PRICING, PRICING_ASOF
from router_benchmark.live.trace_logger import TraceLogger, set_active_trace_logger
from router_benchmark.metrics import (
    compute_pareto_frontier,
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots

LIVE_OUTPUT_ROOT = Path(__file__).parent.parent / "output" / "live"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_live_phase(
    phase_name: str,
    routers: list,
    benchmarks: list,
    seed: int = 1234,
    n_trials: int = 2,
    extra_manifest: dict | None = None,
    resume: bool = False,
) -> Path:
    """Run one live phase (routers x benchmarks) with real API calls.

    resume=False (default): a fresh run. Any pre-existing
    results_incremental.csv / traces.jsonl for this phase are truncated so
    the phase's outputs are built only from this run -- no contamination
    from an earlier aborted attempt.

    resume=True: continue an interrupted run of *this same phase*. Already-
    computed (router, benchmark, task_id, trial) rows in this phase's own
    results_incremental.csv are skipped (no repeat API spend) and the new
    rows are appended to it. The skip-set is derived from this phase's file
    only -- never a hardcoded path to some other phase.
    """
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing required env vars: {missing}")

    out_dir = LIVE_OUTPUT_ROOT / phase_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    manifest = {
        "phase": phase_name,
        "seed": seed,
        "n_trials": n_trials,
        "resume": resume,
        "routers": [r.name for r in routers],
        "benchmarks": [b.name for b in benchmarks],
        "candidate_pricing_usd_per_token": PRICING,
        "pricing_asof": PRICING_ASOF,
        "python_version": sys.version,
        "package_versions": {
            p: _pkg_version(p)
            for p in ["litellm", "semantic-router", "bfcl-eval", "swebench", "openai", "anthropic", "huggingface_hub"]
        },
        **(extra_manifest or {}),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Incremental persistence: every TaskResult is appended (and fsync'd) to
    # results_incremental.csv the instant it's computed, not just held in
    # memory until the whole run finishes. A mid-run crash (OOM kill, etc.)
    # then loses at most the one row in flight, not every row computed so
    # far -- this file is what a resume/recovery pass should read from if
    # `run_live_phase` never reaches the normal results.csv write below.
    incremental_path = out_dir / "results_incremental.csv"
    completed_keys: set = set()
    if resume and incremental_path.exists():
        with open(incremental_path, newline="") as f_done:
            for r in csv.DictReader(f_done):
                completed_keys.add(
                    (r["router_name"], r["benchmark_name"], str(r["task_id"]), str(r.get("trial", "0")))
                )
        print(f"  resume: {len(completed_keys)} completed rows found; skipping those")
    # Fresh run truncates ("w") so this phase's outputs are built only from
    # this run; resume appends ("a") to keep the already-computed rows.
    inc_file = open(incremental_path, "a" if resume else "w", newline="")
    inc_writer = None

    def _on_row_complete(result) -> None:
        nonlocal inc_writer
        row = result.__dict__.copy()
        row["domain"] = row["domain"].value if hasattr(row["domain"], "value") else row["domain"]
        if inc_writer is None:
            import os as _os
            inc_writer = csv.DictWriter(inc_file, fieldnames=list(row.keys()))
            if _os.path.getsize(incremental_path) == 0:
                inc_writer.writeheader()
        inc_writer.writerow(row)
        inc_file.flush()
        import os as _os_2
        _os_2.fsync(inc_file.fileno())

    tracer = TraceLogger(out_dir / "traces.jsonl", append=resume)
    set_active_trace_logger(tracer)
    try:
        print(f"LIVE Phase '{phase_name}': {len(routers)} routers x {len(benchmarks)} benchmarks (real API calls)")
        harness = EvaluationHarness(seed=seed, n_trials=n_trials)
        harness.evaluate(
            routers, benchmarks, on_row_complete=_on_row_complete, completed_keys=completed_keys
        )
        import pandas as pd
        # Read from the incremental file (single source of truth): it holds
        # both any resumed rows and the rows just computed. On a fresh run it
        # was truncated at open, so it contains exactly this run's rows.
        results_df = pd.read_csv(incremental_path)
        print(f"  -> {len(results_df):,} task-trial rows")
    finally:
        tracer.close()
        set_active_trace_logger(None)
        inc_file.close()

    results_df.to_csv(out_dir / "results.csv", index=False)

    per_bm_df = compute_router_benchmark_metrics(results_df)
    per_bm_df.to_csv(out_dir / "metrics_per_benchmark.csv", index=False)

    overall_df = compute_router_overall_metrics(results_df)
    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False)

    frontier_df = compute_pareto_frontier(overall_df)
    frontier_df.to_csv(out_dir / "pareto_frontier.csv", index=False)

    total_cost = results_df["cost_usd"].sum()
    print(f"\nTotal real API spend this phase: ${total_cost:.4f}")
    print(f"Full request/response traces: {out_dir / 'traces.jsonl'}")

    print(f"\n=== LIVE ranking, phase '{phase_name}' (mean success rate) ===")
    cols = ["router_name", "mean_success_rate", "mean_cost_per_task_usd", "mean_latency_p50_ms"]
    print(overall_df[cols].to_string(index=False))

    generate_all_plots(results_df, per_bm_df, overall_df, plots_dir, label=f"live evaluation: {phase_name}")
    print(f"\nAll output for this phase written under: {out_dir}")
    return out_dir
