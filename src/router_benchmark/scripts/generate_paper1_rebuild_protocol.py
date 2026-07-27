#!/usr/bin/env python3
"""Freeze Paper 1 task IDs into the rebuild protocol without reading outcomes.

Only the historical result file's benchmark/task columns are read.  Historical
outcomes remain excluded from the canonical record by the generated protocol.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

from router_benchmark.scripts._paths import repository_root

ROOT = repository_root()
SOURCE = ROOT / "output/live/paper1_live_v3/results.csv"
DESTINATION = ROOT / "protocol/paper1_rebuild.yaml"

ROUTERS = [
    "LiteLLM Router (live)",
    "Aurelio Semantic Router (live)",
    "RouteLLM (live)",
    "vLLM Semantic Router (live)",
]
BENCHMARKS = ["RouterBench (live)", "BFCL v4 (live)", "tau2-bench (live)", "WebArena (live)"]


def digest(task_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


def main() -> None:
    historical = pd.read_csv(SOURCE, usecols=["benchmark_name", "task_id"])
    inventory: dict[str, dict[str, object]] = {}
    for benchmark in BENCHMARKS:
        task_ids = sorted(historical.loc[historical["benchmark_name"] == benchmark, "task_id"].unique().tolist())
        if not task_ids:
            raise SystemExit(f"Historical scope inventory has no IDs for {benchmark}")
        inventory[benchmark] = {
            "subset_id": f"paper1-rebuild-{benchmark.lower().replace(' ', '-').replace('(', '').replace(')', '')}-v1",
            "task_ids": task_ids,
            "task_id_sha256": digest(task_ids),
            "router_trials_per_task": 2,
            "routing_seed_count": 2,
            "outcome_replicates_per_task_candidate": 3,
            "total_route_rows": len(task_ids) * 2 * len(ROUTERS),
            "total_outcome_rows": len(task_ids) * 3 * 3,
        }
    protocol = {
        "protocol_id": "paper1-rebuild-v1",
        "study_status": "frozen_scope_pending_budget_approval",
        "routers": ROUTERS,
        "candidates": ["cheap-small", "mid-general", "strong-frontier"],
        "calibration_policy": "out_of_the_box_only",
        "cross_benchmark_decision": "weighted_success_reported_separately_from_cost",
        "baselines": [
            {
                "name": "Always-Cheapest Baseline (live)", "candidate_policy": "always cheap-small",
                "randomization": "deterministic", "routing_seeds": [], "cascade_order": [],
                "stopping_rule": "one selected candidate per task", "fallback_behavior": "none",
            },
            {
                "name": "Always-Strongest Baseline (live)", "candidate_policy": "always strong-frontier",
                "randomization": "deterministic", "routing_seeds": [], "cascade_order": [],
                "stopping_rule": "one selected candidate per task", "fallback_behavior": "none",
            },
        ],
        "difficulty_band_policy": {
            "method": "task-count terciles within each benchmark",
            "tied_boundary_policy": "sort equal difficulty values by task_id and assign by rank",
        },
        "analysis_paired_comparisons": [list(pair) for pair in combinations(sorted(ROUTERS), 2)],
        "exclusions": [
            "All historical router-specific outcomes, including paper1_live_v3, are audit-only and are not eligible canonical outcomes.",
            "Cache-derived outcomes and router-order-dependent reruns are ineligible.",
            "Routers and benchmarks outside the declared four-by-four scope are ineligible.",
        ],
        "pricing": {"as_of": "2026-07-02", "snapshot_required_for_execution": True},
        "model_snapshots": {
            "cheap-small": "gpt-5.4-nano; execution snapshot must be recorded before run",
            "mid-general": "claude-sonnet-4-6; execution snapshot must be recorded before run",
            "strong-frontier": "claude-opus-4-8; execution snapshot must be recorded before run",
        },
        "grader_versions": {
            "RouterBench (live)": "task-specific exact-match grader; version digest required before run",
            "BFCL v4 (live)": "bfcl-eval==2026.3.23",
            "tau2-bench (live)": "tau2-bench harness revision and image digest required before run",
            "WebArena (live)": "WebArena evaluator revision and environment image digests required before run",
        },
        "benchmarks": inventory,
        "execution_budget": {
            "estimated_api_usd": "UNCONFIRMED",
            "estimated_infrastructure_usd": "UNCONFIRMED",
            "estimated_wall_time": "UNCONFIRMED",
            "stopping_rule": "Stop on any integrity failure; do not replace a missing cell with a router-specific rerun.",
            "approval_status": "pending_author_approval",
        },
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    print(f"Wrote {DESTINATION}")


if __name__ == "__main__":
    main()
