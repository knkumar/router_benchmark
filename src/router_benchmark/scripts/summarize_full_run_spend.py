#!/usr/bin/env python3
"""Summarize actual metered spend from a locked full-run bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _money(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return parsed


def _sum(rows: list[dict[str, str]], field: str, source: str) -> float:
    return sum(_money(row.get(field, ""), f"{source}.{field}") for row in rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"{path} cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_spend(bundle: Path, *, json_output: Path, benchmark_output: Path) -> dict[str, Any]:
    candidate_rows = _read_csv(bundle / "candidate_outcomes.csv")
    route_rows = _read_csv(bundle / "routes.csv")
    provenance = _read_json(bundle / "provenance.json")
    manifest = _read_json(bundle / "manifest.json")

    candidate_model_api_usd = _sum(candidate_rows, "model_api_cost_usd", "candidate_outcomes.csv")
    provider_generation_usd = _sum(candidate_rows, "provider_generation_usd", "candidate_outcomes.csv")
    fallback_generation_usd = _sum(candidate_rows, "fallback_generation_usd", "candidate_outcomes.csv")
    router_service_usd = _sum(route_rows, "router_service_usd", "routes.csv")
    external_metered_usd = _money(provenance.get("external_metered_usd", 0.0), "provenance.external_metered_usd")
    infrastructure_usd = _money(provenance.get("infrastructure_usd", 0.0), "provenance.infrastructure_usd")
    total_usd = candidate_model_api_usd + router_service_usd + external_metered_usd + infrastructure_usd

    by_benchmark: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "benchmark_id": "",
            "candidate_rows": 0,
            "route_rows": 0,
            "candidate_model_api_usd": 0.0,
            "provider_generation_usd": 0.0,
            "fallback_generation_usd": 0.0,
            "router_service_usd": 0.0,
        }
    )
    for row in candidate_rows:
        entry = by_benchmark[row["benchmark_id"]]
        entry["benchmark_id"] = row["benchmark_id"]
        entry["candidate_rows"] += 1
        entry["candidate_model_api_usd"] += _money(row["model_api_cost_usd"], "candidate_outcomes.csv.model_api_cost_usd")
        entry["provider_generation_usd"] += _money(row["provider_generation_usd"], "candidate_outcomes.csv.provider_generation_usd")
        entry["fallback_generation_usd"] += _money(row["fallback_generation_usd"], "candidate_outcomes.csv.fallback_generation_usd")
    for row in route_rows:
        entry = by_benchmark[row["benchmark_id"]]
        entry["benchmark_id"] = row["benchmark_id"]
        entry["route_rows"] += 1
        entry["router_service_usd"] += _money(row["router_service_usd"], "routes.csv.router_service_usd")

    benchmark_rows = []
    for benchmark in sorted(by_benchmark):
        entry = by_benchmark[benchmark]
        candidate_spend = float(entry["candidate_model_api_usd"])
        router_spend = float(entry["router_service_usd"])
        benchmark_rows.append({
            "benchmark_id": benchmark,
            "candidate_rows": entry["candidate_rows"],
            "route_rows": entry["route_rows"],
            "candidate_model_api_usd": f"{candidate_spend:.10f}",
            "provider_generation_usd": f"{float(entry['provider_generation_usd']):.10f}",
            "fallback_generation_usd": f"{float(entry['fallback_generation_usd']):.10f}",
            "router_service_usd": f"{router_spend:.10f}",
            "total_without_external_usd": f"{candidate_spend + router_spend:.10f}",
        })

    provenance_checks = {
        "candidate_model_api_usd_difference": candidate_model_api_usd
        - _money(provenance.get("candidate_model_api_usd", candidate_model_api_usd), "provenance.candidate_model_api_usd"),
        "router_service_usd_difference": router_service_usd
        - _money(provenance.get("router_service_usd", router_service_usd), "provenance.router_service_usd"),
    }
    report = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle),
        "protocol_id": manifest.get("protocol_id"),
        "candidate_rows": len(candidate_rows),
        "route_rows": len(route_rows),
        "candidate_model_api_usd": candidate_model_api_usd,
        "provider_generation_usd": provider_generation_usd,
        "fallback_generation_usd": fallback_generation_usd,
        "router_service_usd": router_service_usd,
        "external_metered_usd": external_metered_usd,
        "infrastructure_usd": infrastructure_usd,
        "total_metered_usd": total_usd,
        "provenance_checks": provenance_checks,
        "benchmark_spend_csv": str(benchmark_output),
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(benchmark_output, benchmark_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_spend(args.bundle, json_output=args.json_output, benchmark_output=args.benchmark_output)
    print(
        "Full-run spend summary written: "
        f"total=${report['total_metered_usd']:.2f}; "
        f"candidate=${report['candidate_model_api_usd']:.2f}; "
        f"router=${report['router_service_usd']:.2f}; "
        f"external=${report['external_metered_usd']:.2f}."
    )


if __name__ == "__main__":
    main()

