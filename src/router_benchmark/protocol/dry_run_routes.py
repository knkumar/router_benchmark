"""Router replay wrapper for the bounded diagnostic dry run.

This command consumes an already completed staged candidate matrix, constructs
frozen benchmark tasks, invokes only router ``route()`` calls, and writes
canonical ``routes.csv`` plus ``router_configs.json`` into the stage directory.
It never scores candidates.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from router_benchmark.interfaces import Benchmark, Router
from router_benchmark.protocol.candidate_runner import CANDIDATE_KEY
from router_benchmark.protocol.dry_run_execution import AdapterFactory
from router_benchmark.protocol.protocol_tools import load_yaml, validate_rebuild_protocol
from router_benchmark.protocol.router_replay import replay_routes
from router_benchmark.scripts.preflight_dry_run import (
    external_metered_reservation_from_protocol,
    router_service_reservation_from_protocol,
    validate_dry_run_protocol,
)


class RouterFactory(Protocol):
    def __call__(self, dry_protocol: Mapping[str, Any]) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Router]]: ...


def _load_factory(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory specs must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError("factory spec must name a callable")
    return factory


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_candidate_keys(protocol: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    return {
        (benchmark, str(task_id), candidate, str(replicate))
        for benchmark, entry in protocol["benchmarks"].items()
        for task_id in entry["task_ids"]
        for candidate in protocol["candidates"]
        for replicate in range(entry["outcome_replicates_per_task_candidate"])
    }


def _external_spend(stage_dir: Path) -> float:
    path = stage_dir / "external_metered_spend.jsonl"
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += float(json.loads(line).get("external_metered_usd", 0.0) or 0.0)
    return total


def _assert_completed_candidate_stage(stage_dir: Path, dry_protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    rows_path = stage_dir / "candidate_outcomes.csv"
    manifest_path = stage_dir / "stage_manifest.json"
    traces_path = stage_dir / "traces.jsonl"
    if not rows_path.exists() or not manifest_path.exists() or not traces_path.exists():
        raise ValueError("candidate stage must contain candidate_outcomes.csv, traces.jsonl, and stage_manifest.json")
    rows = _read_csv(rows_path)
    keys = {tuple(row[field] for field in CANDIDATE_KEY) for row in rows}
    if keys != _expected_candidate_keys(dry_protocol):
        raise ValueError("candidate stage is incomplete or does not match dry-run protocol")
    if any(row.get("cache_flag", "").lower() != "false" for row in rows):
        raise ValueError("candidate stage contains cache-derived rows")
    for row in rows:
        values = [
            float(row["provider_generation_usd"]),
            float(row["fallback_generation_usd"]),
            float(row["model_api_cost_usd"]),
            float(row["generation_latency_ms"]),
        ]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("candidate stage contains nonfinite or negative cost/latency")
        if abs(values[2] - values[0] - values[1]) > 1e-9:
            raise ValueError("candidate stage has invalid named candidate cost components")
    return rows


def _tasks_by_benchmark(protocol: Mapping[str, Any], adapters: Mapping[str, Benchmark]):
    return {
        benchmark_id: list(adapters[benchmark_id].generate_tasks(rng=None))
        for benchmark_id in protocol["benchmarks"]
    }


def run_dry_route_stage(
    dry_protocol: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    *,
    stage_dir: Path,
    adapters: Mapping[str, Benchmark],
    router_configs: Mapping[str, Mapping[str, Any]],
    routers: Mapping[str, Router],
    overwrite: bool = False,
) -> list[dict[str, str]]:
    validate_rebuild_protocol(frozen_protocol)
    validate_dry_run_protocol(dict(dry_protocol), dict(frozen_protocol))
    stage_dir = Path(stage_dir)
    candidate_rows = _assert_completed_candidate_stage(stage_dir, dry_protocol)
    routes_path = stage_dir / "routes.csv"
    router_configs_path = stage_dir / "router_configs.json"
    if not overwrite and (routes_path.exists() or router_configs_path.exists()):
        raise ValueError("route stage outputs already exist; refusing to rerun without --overwrite")
    tasks = _tasks_by_benchmark(dry_protocol, adapters)
    rows = replay_routes(
        dry_protocol,
        router_configs=router_configs,
        routers=routers,
        tasks_by_benchmark=tasks,
    )
    service_spend = sum(float(row["router_service_usd"]) for row in rows)
    service_reserved = router_service_reservation_from_protocol(dict(dry_protocol))
    if service_spend > service_reserved + 1e-9:
        raise RuntimeError("router-service spend exceeds its dry-run reservation")
    candidate_spend = sum(float(row["model_api_cost_usd"]) for row in candidate_rows)
    external_spend = _external_spend(stage_dir)
    if external_spend > external_metered_reservation_from_protocol(dict(dry_protocol)) + 1e-9:
        raise RuntimeError("external metered spend exceeds its dry-run reservation")
    if candidate_spend + external_spend + service_spend > float(dry_protocol["dry_run_budget_cap_usd"]) + 1e-9:
        raise RuntimeError("dry-run measured spend exceeds budget cap")

    with routes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    router_configs_path.write_text(json.dumps(dict(router_configs), indent=2, default=str) + "\n", encoding="utf-8")
    return rows


def _assert_required_metered_environment() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("missing required provider environment variable: OPENAI_API_KEY")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-protocol", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--router-factory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    dry_protocol = load_yaml(args.dry_protocol)
    frozen_protocol = load_yaml(args.frozen_protocol)
    validate_rebuild_protocol(frozen_protocol)
    validate_dry_run_protocol(dry_protocol, frozen_protocol)
    _assert_required_metered_environment()
    adapters = _load_factory(args.adapter_factory)(dry_protocol)
    router_configs, routers = _load_factory(args.router_factory)(dry_protocol)
    rows = run_dry_route_stage(
        dry_protocol,
        frozen_protocol,
        stage_dir=args.stage_dir,
        adapters=adapters,
        router_configs=router_configs,
        routers=routers,
        overwrite=args.overwrite,
    )
    spend = sum(float(row["router_service_usd"]) for row in rows)
    print(f"Dry route stage wrote {len(rows)} rows to {args.stage_dir}; router_service_usd={spend:.6f}.")


if __name__ == "__main__":
    main()
