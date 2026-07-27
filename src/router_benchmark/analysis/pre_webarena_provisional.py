#!/usr/bin/env python3
"""Summarize the completed non-WebArena candidate matrix and route replay.

This script is deliberately separate from canonical analysis.  It writes
descriptive interim artifacts only and rejects a stage containing WebArena
rows.  The full paper rebuild must use the locked canonical bundle instead.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


BENCHMARKS = ("RouterBench (live)", "BFCL v4 (live)", "tau2-bench (live)")
CANDIDATES = ("cheap-small", "mid-general", "strong-frontier")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"{path.name} has no rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    samples = np.empty(draws)
    for draw in range(draws):
        samples[draw] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(stage_dir: Path, output_dir: Path, *, draws: int = 10_000, seed: int = 20_260_721) -> None:
    candidates = _read_csv(stage_dir / "candidate_outcomes.csv")
    routes = _read_csv(stage_dir / "routes.csv")
    configs = json.loads((stage_dir / "router_configs.json").read_text(encoding="utf-8"))
    observed_benchmarks = {row["benchmark_id"] for row in candidates} | {row["benchmark_id"] for row in routes}
    if observed_benchmarks != set(BENCHMARKS):
        raise ValueError(f"provisional stage must contain exactly {BENCHMARKS}; observed={sorted(observed_benchmarks)}")
    expected_candidate_rows = 1710
    expected_route_rows = 1520
    if len(candidates) != expected_candidate_rows or len(routes) != expected_route_rows:
        raise ValueError(
            f"provisional stage has candidate_rows={len(candidates)} and route_rows={len(routes)}; "
            f"expected {expected_candidate_rows} and {expected_route_rows}"
        )
    route_keys = {(row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"]) for row in routes}
    if len(route_keys) != len(routes):
        raise ValueError("routes.csv contains duplicate primary keys")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_index = {
        (row["benchmark_id"], row["task_id"], row["candidate_id"], row["outcome_replicate"]): row
        for row in candidates
    }
    candidate_summary = []
    for benchmark in BENCHMARKS:
        for candidate in CANDIDATES:
            rows = [row for row in candidates if row["benchmark_id"] == benchmark and row["candidate_id"] == candidate]
            candidate_summary.append(
                {
                    "benchmark_id": benchmark,
                    "candidate_id": candidate,
                    "candidate_rows": len(rows),
                    "success_rate": _mean(row["success"].lower() == "true" for row in rows),
                    "model_api_cost_usd_mean": _mean(float(row["model_api_cost_usd"]) for row in rows),
                }
            )
    _write_csv(output_dir / "candidate_tier_summary.csv", candidate_summary)

    task_scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    selected: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    router_summary = []
    grouped: dict[tuple[str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for route in routes:
        router_name = configs[route["router_config_id"]]["router_name"]
        group = (route["benchmark_id"], router_name)
        selected[group][route["selected_candidate"]] += 1
        for replicate in range(3):
            candidate = candidate_index[(route["benchmark_id"], route["task_id"], route["selected_candidate"], str(replicate))]
            grouped[group].append((route, candidate))
            task_scores[(route["benchmark_id"], route["task_id"])][router_name].append(
                float(candidate["success"].lower() == "true")
            )

    router_names = [configs[router_id]["router_name"] for router_id in configs]
    rng = np.random.default_rng(seed)
    for benchmark in BENCHMARKS:
        tasks = sorted(task for observed_benchmark, task in task_scores if observed_benchmark == benchmark)
        for router_name in router_names:
            rows = grouped[(benchmark, router_name)]
            task_values = np.array([_mean(task_scores[(benchmark, task)][router_name]) for task in tasks])
            low, high = _bootstrap_interval(task_values, rng, draws)
            router_summary.append(
                {
                    "benchmark_id": benchmark,
                    "router_name": router_name,
                    "joined_outcomes": len(rows),
                    "success_rate": float(task_values.mean()),
                    "success_ci_low": low,
                    "success_ci_high": high,
                    "candidate_model_api_usd_mean": _mean(float(candidate["model_api_cost_usd"]) for _, candidate in rows),
                    "router_service_usd_mean": _mean(float(route["router_service_usd"]) for route, _ in rows),
                    "cheap_routes": selected[(benchmark, router_name)]["cheap-small"],
                    "mid_routes": selected[(benchmark, router_name)]["mid-general"],
                    "strong_routes": selected[(benchmark, router_name)]["strong-frontier"],
                }
            )
    _write_csv(output_dir / "router_summary.csv", router_summary)

    pairwise = []
    for benchmark in BENCHMARKS:
        tasks = sorted(task for observed_benchmark, task in task_scores if observed_benchmark == benchmark)
        router_values = {
            router: np.array([_mean(task_scores[(benchmark, task)][router]) for task in tasks])
            for router in router_names
        }
        for index, router_1 in enumerate(router_names):
            for router_2 in router_names[index + 1 :]:
                differences = router_values[router_1] - router_values[router_2]
                low, high = _bootstrap_interval(differences, rng, draws)
                pairwise.append(
                    {
                        "benchmark_id": benchmark,
                        "router_1": router_1,
                        "router_2": router_2,
                        "risk_difference": float(differences.mean()),
                        "risk_difference_ci_low": low,
                        "risk_difference_ci_high": high,
                        "tasks": len(tasks),
                    }
                )
    _write_csv(output_dir / "router_pairwise_differences.csv", pairwise)
    (output_dir / "README.md").write_text(
        "# Provisional Non-WebArena Analysis\n\n"
        "This directory is descriptive interim evidence only. It excludes WebArena and is not a canonical bundle or submission artifact. "
        "Intervals use task-clustered bootstrap resampling with the recorded seed and draw count.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    args = parser.parse_args()
    if args.draws < 1:
        raise ValueError("draws must be positive")
    summarize(args.stage_dir, args.output_dir, draws=args.draws, seed=args.seed)


if __name__ == "__main__":
    main()

