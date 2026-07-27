"""Route replay wrapper for the approved full Paper 1 rebuild."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from router_benchmark.protocol.dry_run_routes import (
    _assert_completed_candidate_stage,
    _external_spend,
    _load_factory,
    _tasks_by_benchmark,
)
from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.protocol.router_replay import replay_routes
from router_benchmark.scripts.preflight_full_run import (
    external_metered_reservation_from_full_protocol,
    router_service_reservation_from_full_protocol,
    validate_full_run_protocol,
)


def run_full_route_stage(
    protocol: Mapping[str, Any],
    *,
    stage_dir: Path,
    adapters: Mapping[str, object],
    router_configs: Mapping[str, Mapping[str, Any]],
    routers: Mapping[str, object],
    overwrite: bool = False,
    exclude_benchmark: str | None = None,
    include_benchmark: str | None = None,
) -> list[dict[str, str]]:
    validate_full_run_protocol(dict(protocol))
    if exclude_benchmark and include_benchmark:
        raise ValueError("exclude_benchmark and include_benchmark are mutually exclusive")
    effective_protocol = dict(protocol)
    if exclude_benchmark:
        effective_protocol["benchmarks"] = {
            name: entry for name, entry in protocol["benchmarks"].items()
            if name != exclude_benchmark
        }
    if include_benchmark:
        if include_benchmark not in protocol["benchmarks"]:
            raise ValueError(f"unknown benchmark: {include_benchmark}")
        effective_protocol["benchmarks"] = {
            include_benchmark: protocol["benchmarks"][include_benchmark]
        }
    stage_dir = Path(stage_dir)
    candidate_rows = _assert_completed_candidate_stage(stage_dir, effective_protocol)
    routes_path = stage_dir / "routes.csv"
    router_configs_path = stage_dir / "router_configs.json"
    if not overwrite and (routes_path.exists() or router_configs_path.exists()):
        raise ValueError("route stage outputs already exist; refusing to rerun without --overwrite")
    tasks = _tasks_by_benchmark(effective_protocol, adapters)  # type: ignore[arg-type]
    rows = replay_routes(
        effective_protocol,
        router_configs=router_configs,
        routers=routers,  # type: ignore[arg-type]
        tasks_by_benchmark=tasks,
    )
    service_spend = sum(float(row["router_service_usd"]) for row in rows)
    service_reserved = router_service_reservation_from_full_protocol(protocol)
    if service_spend > service_reserved + 1e-9:
        raise RuntimeError("router-service spend exceeds its full-run reservation")
    candidate_spend = sum(float(row["model_api_cost_usd"]) for row in candidate_rows)
    external_spend = _external_spend(stage_dir)
    if external_spend > external_metered_reservation_from_full_protocol(protocol) + 1e-9:
        raise RuntimeError("external metered spend exceeds its full-run reservation")
    if candidate_spend + external_spend + service_spend > float(protocol["full_run_cost_reservations"]["total_cap_usd"]) + 1e-9:
        raise RuntimeError("full-run measured spend exceeds budget cap")

    with routes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    router_configs_path.write_text(json.dumps(dict(router_configs), indent=2, default=str) + "\n", encoding="utf-8")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--router-factory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exclude-benchmark")
    parser.add_argument("--include-benchmark")
    args = parser.parse_args(argv)
    protocol = load_yaml(args.protocol)
    validate_full_run_protocol(protocol)
    adapters = _load_factory(args.adapter_factory)(protocol)
    router_configs, routers = _load_factory(args.router_factory)(protocol)
    rows = run_full_route_stage(
        protocol,
        stage_dir=args.stage_dir,
        adapters=adapters,
        router_configs=router_configs,
        routers=routers,
        overwrite=args.overwrite,
        exclude_benchmark=args.exclude_benchmark,
        include_benchmark=args.include_benchmark,
    )
    print(f"Full route stage wrote {len(rows)} rows to {args.stage_dir}.")


if __name__ == "__main__":
    main()
