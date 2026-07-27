#!/usr/bin/env python3
"""Prespecified paired effects from a locked canonical bundle only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.protocol.canonical import validate_bundle
from router_benchmark.protocol.protocol_tools import load_yaml, validate_analysis_protocol
from router_benchmark.analysis.resampling import paired_success_difference_test, paired_task_replicate_draws


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_analysis_protocol_path() -> Path:
    """Locate the versioned analysis protocol for source and wheel installs.

    Source checkouts keep it at the repository root. Container images copy it
    into the working directory while installing the package into site-packages.
    Callers outside either layout must pass ``analysis_protocol_path``.
    """
    candidates = (Path.cwd() / "protocol" / "analysis.yaml", ROOT / "protocol" / "analysis.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _holm(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm adjustment without an undeclared analysis dependency."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    rejected = [False] * count
    running = 0.0
    still_rejecting = True
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
        if p_values[index] > alpha / (count - rank):
            still_rejecting = False
        rejected[index] = still_rejecting
    return rejected, adjusted


def paired_effects(
    bundle_dir: Path, protocol_path: Path, analysis_protocol_path: Path | None = None
) -> tuple[list[dict[str, object]], dict[str, list[tuple[tuple[str, int], ...]]]]:
    validate_bundle(bundle_dir, protocol_path)
    protocol = load_yaml(protocol_path)
    analysis_protocol_path = analysis_protocol_path or _default_analysis_protocol_path()
    analysis = load_yaml(analysis_protocol_path)
    validate_analysis_protocol(analysis)
    config = json.loads((bundle_dir / "router_configs.json").read_text(encoding="utf-8"))
    router_names = {config_id: value["router_name"] for config_id, value in config.items()}
    declared = set(protocol["routers"])
    if set(router_names.values()) != declared:
        raise ValueError("router_configs.json must map each config ID to one declared router")
    name_to_config = {name: config_id for config_id, name in router_names.items()}

    candidate = {
        (row["benchmark_id"], row["task_id"], row["candidate_id"], row["outcome_replicate"]): row
        for row in _read_csv(bundle_dir / "candidate_outcomes.csv")
    }
    routes = {
        (row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"]): row
        for row in _read_csv(bundle_dir / "routes.csv")
    }
    observations: dict[tuple[str, str, str, str, str], bool] = {}
    for row in _read_csv(bundle_dir / "outcomes.csv"):
        route_key = (row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"])
        route = routes[route_key]
        candidate_key = (route["benchmark_id"], route["task_id"], route["selected_candidate"], row["outcome_replicate"])
        observations[(route["router_config_id"], route["benchmark_id"], route["task_id"], route["routing_seed"], row["outcome_replicate"])] = candidate[candidate_key]["success"].lower() == "true"

    requested = {tuple(pair) for pair in protocol["analysis_paired_comparisons"]}
    rows: list[dict[str, object]] = []
    saved_draws: dict[str, list[tuple[tuple[str, int], ...]]] = {}
    for benchmark in protocol["benchmarks"]:
        benchmark_entry = protocol["benchmarks"][benchmark]
        draws = paired_task_replicate_draws(
            benchmark_entry["task_ids"],
            outcome_replicates=benchmark_entry["outcome_replicates_per_task_candidate"],
            draws=analysis["resampling"]["draws"],
            seed=analysis["resampling"]["seed"],
        )
        saved_draws[benchmark] = [draw.task_replicates for draw in draws]
        for first_name, second_name in sorted(requested):
            first = name_to_config[first_name]
            second = name_to_config[second_name]
            # Average every routing trial and outcome replicate nested inside a
            # task down to one per-task success rate per router. This is the
            # cluster the bootstrap and the discordant counts operate on, so the
            # inferential unit is the task, not the joined row.
            averaged_outcomes: dict[tuple[str, str, int], list[float]] = {}
            per_task: dict[str, dict[str, list[float]]] = {}
            for (router_id, observed_benchmark, task_id, _seed, replicate), success in observations.items():
                if observed_benchmark == benchmark and router_id in {first, second}:
                    averaged_outcomes.setdefault((router_id, task_id, int(replicate)), []).append(float(success))
                    per_task.setdefault(task_id, {}).setdefault(router_id, []).append(float(success))
            outcome_means = {key: sum(values) / len(values) for key, values in averaged_outcomes.items()}
            risk_difference, ci_low, ci_high, raw_p_value = paired_success_difference_test(
                outcome_means, first_router=first, second_router=second, draws=draws
            )
            # Task-level discordant counts: tasks where one router's mean success
            # over its nested trials/replicates strictly exceeds the other's.
            n10 = sum(1 for record in per_task.values()
                      if sum(record[first]) / len(record[first]) > sum(record[second]) / len(record[second]))
            n01 = sum(1 for record in per_task.values()
                      if sum(record[first]) / len(record[first]) < sum(record[second]) / len(record[second]))
            n_tasks = len(per_task)
            rows.append({
                "benchmark_id": benchmark, "router_1": first_name, "router_2": second_name, "paired_n": n_tasks,
                "n01": n01, "n10": n10, "risk_difference": risk_difference,
                "risk_difference_ci_low": ci_low, "risk_difference_ci_high": ci_high,
                "test": "task_cluster_bootstrap", "raw_p_value": raw_p_value,
                "rebuild_protocol_sha256": _protocol_hash(protocol_path),
                "analysis_protocol_sha256": _protocol_hash(analysis_protocol_path),
            })
    for benchmark in protocol["benchmarks"]:
        group = [row for row in rows if row["benchmark_id"] == benchmark]
        rejected, adjusted = _holm([row["raw_p_value"] for row in group])
        for row, adjusted_p, reject in zip(group, adjusted, rejected):
            row["adjusted_p_value"] = adjusted_p
            row["correction_family"] = "six router-pair comparisons within benchmark"
            row["reject_null"] = bool(reject)
    return rows, saved_draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, default=_default_analysis_protocol_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws-output", type=Path, required=True)
    args = parser.parse_args()
    rows, draws = paired_effects(args.bundle, args.protocol, args.analysis_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.draws_output.parent.mkdir(parents=True, exist_ok=True)
    args.draws_output.write_text(json.dumps(draws, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
