"""Write one locked canonical bundle from already-executed outcome and route rows.

This module deliberately has no benchmark or provider imports.  Candidate
execution and router replay supply rows; this writer makes their lineage,
checksums, and lock state reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from router_benchmark.protocol.canonical import REQUIRED_FILES


def trace_digest(record: dict[str, Any]) -> str:
    """Return the content digest used by canonical trace records."""
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name} cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_locked_bundle(
    bundle_dir: Path,
    protocol: dict[str, Any],
    *,
    router_configs: dict[str, dict[str, Any]],
    candidate_outcomes: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    traces: Iterable[dict[str, Any]],
    provenance: dict[str, Any],
) -> Path:
    """Write and checksum a complete bundle after all outcome and route work ends."""
    bundle_dir.mkdir(parents=True, exist_ok=False)
    trace_rows = []
    for record in traces:
        record = dict(record)
        record.setdefault("digest", trace_digest(record))
        trace_rows.append(record)

    outcomes = []
    for route in routes:
        for replicate in range(protocol["benchmarks"][route["benchmark_id"]]["outcome_replicates_per_task_candidate"]):
            key = "|".join((route["benchmark_id"], route["task_id"], route["selected_candidate"], str(replicate)))
            outcomes.append({**{field: route[field] for field in ("router_config_id", "benchmark_id", "task_id", "routing_seed")}, "outcome_replicate": str(replicate), "candidate_outcome_key": key})

    counts = {
        benchmark: {
            "unique_tasks": len(entry["task_ids"]),
            "router_trials_per_task": entry["router_trials_per_task"],
            "routing_seed_count": entry["routing_seed_count"],
            "outcome_replicates_per_task_candidate": entry["outcome_replicates_per_task_candidate"],
            "total_route_rows": entry["total_route_rows"],
            "total_outcome_rows": entry["total_outcome_rows"],
            "subset_id": entry["subset_id"],
        }
        for benchmark, entry in protocol["benchmarks"].items()
    }
    (bundle_dir / "manifest.json").write_text(json.dumps({"protocol_id": protocol["protocol_id"], "bundle_status": "locked", "router_config_ids": list(router_configs), "benchmark_counts": counts}, indent=2), encoding="utf-8")
    (bundle_dir / "router_configs.json").write_text(json.dumps(router_configs, indent=2), encoding="utf-8")
    _write_csv(bundle_dir / "candidate_outcomes.csv", candidate_outcomes)
    _write_csv(bundle_dir / "routes.csv", routes)
    _write_csv(bundle_dir / "outcomes.csv", outcomes)
    _write_csv(bundle_dir / "results.csv", outcomes)
    (bundle_dir / "traces.jsonl").write_text("".join(json.dumps(row, default=str) + "\n" for row in trace_rows), encoding="utf-8")
    (bundle_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    checksums = {name: hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest() for name in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle_dir / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    return bundle_dir
