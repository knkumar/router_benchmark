#!/usr/bin/env python3
"""Merge retained full-stage routes with replacement WebArena route rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.preflight_full_run import validate_full_run_protocol


WEBARENA = "WebArena (live)"
ROUTE_KEY = ("router_config_id", "benchmark_id", "task_id", "routing_seed")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_routes(*, source_stage: Path, replacement_stage: Path, protocol: dict) -> None:
    validate_full_run_protocol(protocol)
    replacement_path = replacement_stage / "routes.csv"
    source_path = source_stage / "routes.csv"
    if not source_path.exists() or not replacement_path.exists():
        raise ValueError("both source and replacement route files are required")
    retained = [row for row in _read_rows(source_path) if row["benchmark_id"] != WEBARENA]
    replacement = _read_rows(replacement_path)
    if not replacement or any(row["benchmark_id"] != WEBARENA for row in replacement):
        raise ValueError("replacement route file must contain only WebArena rows")
    merged = retained + replacement
    keys = [tuple(row[field] for field in ROUTE_KEY) for row in merged]
    if len(keys) != len(set(keys)):
        raise ValueError("merged route rows contain duplicate primary keys")
    expected = sum(entry["total_route_rows"] for entry in protocol["benchmarks"].values())
    if len(merged) != expected:
        raise ValueError(f"merged route row count is {len(merged)}, expected {expected}")
    with replacement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    lineage_path = replacement_stage / "repair_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {}
    lineage.update({
        "retained_route_source_stage": str(source_stage),
        "replacement_route_rows": len(replacement),
        "retained_route_rows": len(retained),
    })
    lineage_path.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--replacement-stage", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    merge_routes(
        source_stage=args.source_stage,
        replacement_stage=args.replacement_stage,
        protocol=load_yaml(args.protocol),
    )


if __name__ == "__main__":
    main()

