"""Reviewer-facing evidence gates for the Paper 1 canonical rebuild."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.protocol.canonical import validate_bundle
from router_benchmark.protocol.protocol_tools import load_yaml


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"{path.name} cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _candidate_key(row: dict[str, str]) -> str:
    return "|".join((row["benchmark_id"], row["task_id"], row["candidate_id"], row["outcome_replicate"]))


def write_candidate_tier_summary(candidate_rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(row["benchmark_id"], row["candidate_id"])].append(row)
    rows = []
    for (benchmark_id, candidate_id), group in sorted(grouped.items()):
        rows.append({
            "benchmark_id": benchmark_id,
            "candidate_id": candidate_id,
            "candidate_rows": len(group),
            "success_rate": _mean([1.0 if row["success"].lower() == "true" else 0.0 for row in group]),
            "model_api_cost_usd_mean": _mean([float(row["model_api_cost_usd"]) for row in group]),
            "generation_latency_ms_mean": _mean([float(row["generation_latency_ms"]) for row in group]),
        })
    _write_csv(output, rows)


def write_baseline_summary(candidate_rows: list[dict[str, str]], output: Path) -> None:
    baselines = {
        "Always-Cheapest Baseline (live)": "cheap-small",
        "Always-Mid Baseline (live)": "mid-general",
        "Always-Strongest Baseline (live)": "strong-frontier",
    }
    rows = []
    for baseline_id, candidate_id in baselines.items():
        by_benchmark: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidate_rows:
            if row["candidate_id"] == candidate_id:
                by_benchmark[row["benchmark_id"]].append(row)
        for benchmark_id, group in sorted(by_benchmark.items()):
            rows.append({
                "baseline_id": baseline_id,
                "candidate_policy": f"always {candidate_id}",
                "benchmark_id": benchmark_id,
                "candidate_rows": len(group),
                "success_rate": _mean([1.0 if row["success"].lower() == "true" else 0.0 for row in group]),
                "model_api_cost_usd_mean": _mean([float(row["model_api_cost_usd"]) for row in group]),
                "generation_latency_ms_mean": _mean([float(row["generation_latency_ms"]) for row in group]),
                "generation_latency_ms_p50": _median([float(row["generation_latency_ms"]) for row in group]),
            })
    _write_csv(output, rows)


def write_baseline_reconciliation(
    candidate_rows: list[dict[str, str]],
    routes: list[dict[str, str]],
    outcomes: list[dict[str, str]],
    output: Path,
) -> None:
    candidate_by_key = {_candidate_key(row): row for row in candidate_rows}
    route_by_key = {
        (row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"]): row
        for row in routes
    }
    checked = 0
    baseline_tier_matches = 0
    for outcome in outcomes:
        route_key = (
            outcome["router_config_id"],
            outcome["benchmark_id"],
            outcome["task_id"],
            outcome["routing_seed"],
        )
        route = route_by_key[route_key]
        expected_key = "|".join((
            route["benchmark_id"],
            route["task_id"],
            route["selected_candidate"],
            outcome["outcome_replicate"],
        ))
        if outcome["candidate_outcome_key"] != expected_key:
            raise ValueError("route outcome does not reconcile to selected candidate")
        if expected_key not in candidate_by_key:
            raise ValueError("route outcome references missing candidate row")
        checked += 1
        if route["selected_candidate"] in {"cheap-small", "mid-general", "strong-frontier"}:
            baseline_tier_matches += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "passed",
                "checked_joined_outcomes": checked,
                "baseline_tier_route_matches": baseline_tier_matches,
                "deterministic_baselines": [
                    "Always-Cheapest Baseline (live)",
                    "Always-Mid Baseline (live)",
                    "Always-Strongest Baseline (live)",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_bfcl_route_equivalence(
    candidate_rows: list[dict[str, str]],
    routes: list[dict[str, str]],
    protocol: dict[str, Any],
    output: Path,
) -> None:
    candidate_by_key = {_candidate_key(row): row for row in candidate_rows}
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for route in routes:
        if route["benchmark_id"] == "BFCL v4 (live)":
            grouped[(route["task_id"], route["routing_seed"], route["selected_candidate"])].append(route)
    rows = []
    checker = protocol["grader_versions"]["BFCL v4 (live)"]
    for (task_id, routing_seed, selected_candidate), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        for route_a, route_b in itertools.combinations(sorted(group, key=lambda row: row["router_config_id"]), 2):
            for replicate in range(protocol["benchmarks"]["BFCL v4 (live)"]["outcome_replicates_per_task_candidate"]):
                key = f"BFCL v4 (live)|{task_id}|{selected_candidate}|{replicate}"
                candidate = candidate_by_key[key]
                rows.append({
                    "task_id": task_id,
                    "routing_seed": routing_seed,
                    "outcome_replicate": replicate,
                    "router_a": route_a["router_config_id"],
                    "router_b": route_b["router_config_id"],
                    "selected_candidate": selected_candidate,
                    "candidate_outcome_key": key,
                    "success_a": candidate["success"],
                    "success_b": candidate["success"],
                    "model_api_cost_usd_a": candidate["model_api_cost_usd"],
                    "model_api_cost_usd_b": candidate["model_api_cost_usd"],
                    "equivalent_outcome": "true",
                    "checker": "canonical candidate-row equivalence",
                    "checker_version": checker,
                })
    if not rows:
        rows = [{
            "task_id": "none",
            "routing_seed": "none",
            "outcome_replicate": "none",
            "router_a": "none",
            "router_b": "none",
            "selected_candidate": "none",
            "candidate_outcome_key": "none",
            "success_a": "none",
            "success_b": "none",
            "model_api_cost_usd_a": "none",
            "model_api_cost_usd_b": "none",
            "equivalent_outcome": "not_applicable",
            "checker": "canonical candidate-row equivalence",
            "checker_version": checker,
        }]
    _write_csv(output, rows)


def write_ablation_registry(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "deferred",
                "canonical_ablation_claims": [],
                "reason": (
                    "No mechanism-isolating ablation is part of paper1-rebuild-v1; "
                    "price, provider, model identity, metadata, and calibration claims must use a separate protocol."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_reviewer_gates(bundle: Path, protocol_path: Path, output_dir: Path) -> None:
    validate_bundle(bundle, protocol_path)
    protocol = load_yaml(protocol_path)
    candidate_rows = _read_csv(bundle / "candidate_outcomes.csv")
    routes = _read_csv(bundle / "routes.csv")
    outcomes = _read_csv(bundle / "outcomes.csv")
    write_candidate_tier_summary(candidate_rows, output_dir / "candidate_tier_summary.csv")
    write_baseline_summary(candidate_rows, output_dir / "baseline_summary.csv")
    write_baseline_reconciliation(candidate_rows, routes, outcomes, output_dir / "baseline_reconciliation.json")
    write_bfcl_route_equivalence(candidate_rows, routes, protocol, output_dir / "bfcl_route_equivalence.csv")
    write_ablation_registry(output_dir / "ablation_registry.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_reviewer_gates(args.bundle, args.protocol, args.output_dir)
    print(f"Reviewer gate artifacts written to {args.output_dir}.")


if __name__ == "__main__":
    main()
