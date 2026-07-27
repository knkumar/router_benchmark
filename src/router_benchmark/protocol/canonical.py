"""Schema and integrity checks for a locked Paper 1 canonical bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from router_benchmark.protocol.protocol_tools import ProtocolValidationError, load_yaml, validate_rebuild_protocol


REQUIRED_FILES = {
    "manifest.json", "outcomes.csv", "routes.csv", "router_configs.json",
    "candidate_outcomes.csv", "results.csv", "traces.jsonl", "provenance.json", "checksums.sha256",
}
CANDIDATE_KEY = ("benchmark_id", "task_id", "candidate_id", "outcome_replicate")
ROUTE_KEY = ("router_config_id", "benchmark_id", "task_id", "routing_seed")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_columns(rows: list[dict[str, str]], required: set[str], filename: str) -> None:
    if not rows or not rows[0]:
        raise ProtocolValidationError(f"{filename} must contain at least one row")
    missing = required - set(rows[0])
    if missing:
        raise ProtocolValidationError(f"{filename} missing columns {sorted(missing)}")


def _unique_nonempty(rows: Iterable[dict[str, str]], key: tuple[str, ...], filename: str) -> set[tuple[str, ...]]:
    values: set[tuple[str, ...]] = set()
    for row in rows:
        value = tuple(row.get(column, "") for column in key)
        if any(not part for part in value):
            raise ProtocolValidationError(f"{filename} has null primary key {key}")
        if value in values:
            raise ProtocolValidationError(f"{filename} has duplicate primary key {value}")
        values.add(value)
    return values


def _load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        checksums[filename] = digest
    return checksums


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_route_equivalent_outcomes(rows: Iterable[dict[str, str]]) -> None:
    """Reject router-specific outcomes for an identical task/candidate draw.

    This is an audit helper for historical flat result files.  A canonical
    lookup bundle prevents this condition by joining every route to the same
    candidate-outcome key.
    """
    seen: dict[tuple[str, str, str, str], tuple[str, str, str]] = {}
    for row in rows:
        key = (row["benchmark_name"], row["task_id"], row["trial"], row["selected_candidate"])
        outcome = (row["success"], row["cost_usd"], row["latency_ms"])
        previous = seen.setdefault(key, outcome)
        if previous != outcome:
            raise ProtocolValidationError(
                f"identical route decision has router-specific outcome for {key}"
            )


def validate_bundle(bundle_dir: Path, protocol_path: Path) -> None:
    """Validate one canonical bundle without provider calls or reruns."""
    protocol = load_yaml(protocol_path)
    validate_rebuild_protocol(protocol, require_approved_budget=True)
    present = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - present
    if missing:
        raise ProtocolValidationError(f"canonical bundle missing files {sorted(missing)}")
    untracked = present - REQUIRED_FILES
    if untracked:
        raise ProtocolValidationError(f"canonical bundle has untracked files {sorted(untracked)}")

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise ProtocolValidationError("manifest protocol_id does not match protocol")
    if manifest.get("bundle_status") != "locked":
        raise ProtocolValidationError("canonical bundle must be locked")
    manifest_counts = manifest.get("benchmark_counts")
    if not isinstance(manifest_counts, dict) or set(manifest_counts) != set(protocol["benchmarks"]):
        raise ProtocolValidationError("manifest benchmark_counts must cover every declared benchmark")
    for benchmark, entry in protocol["benchmarks"].items():
        counts = manifest_counts[benchmark]
        expected_counts = {
            "unique_tasks": len(entry["task_ids"]),
            "router_trials_per_task": entry["router_trials_per_task"],
            "routing_seed_count": entry["routing_seed_count"],
            "outcome_replicates_per_task_candidate": entry["outcome_replicates_per_task_candidate"],
            "total_route_rows": entry["total_route_rows"],
            "total_outcome_rows": entry["total_outcome_rows"],
            "subset_id": entry["subset_id"],
        }
        if counts != expected_counts:
            raise ProtocolValidationError(f"manifest count mismatch for {benchmark}")

    router_configs = json.loads((bundle_dir / "router_configs.json").read_text(encoding="utf-8"))
    if not isinstance(router_configs, dict):
        raise ProtocolValidationError("router_configs.json must be a mapping")
    declared_router_names = set(protocol["routers"])
    if set(router_configs) != set(manifest.get("router_config_ids", [])):
        raise ProtocolValidationError("router_configs.json IDs must match manifest router_config_ids")
    if {entry.get("router_name") for entry in router_configs.values() if isinstance(entry, dict)} != declared_router_names:
        raise ProtocolValidationError("router_configs.json contains an undeclared router or omits a declared router")

    checksums = _load_checksums(bundle_dir / "checksums.sha256")
    expected_checksum_files = REQUIRED_FILES - {"checksums.sha256"}
    if set(checksums) != expected_checksum_files:
        raise ProtocolValidationError("checksums.sha256 must cover every required file except itself")
    for filename, digest in checksums.items():
        if _sha256(bundle_dir / filename) != digest:
            raise ProtocolValidationError(f"checksum mismatch for {filename}")

    candidate_rows = _read_csv(bundle_dir / "candidate_outcomes.csv")
    _require_columns(candidate_rows, set(CANDIDATE_KEY) | {
        "execution_seed", "grader_version", "raw_trace_digest", "success", "provider_generation_usd",
        "fallback_generation_usd", "model_api_cost_usd", "generation_latency_ms", "failure_status",
        "cache_flag", "model_version", "pricing_snapshot",
    }, "candidate_outcomes.csv")
    candidate_keys = _unique_nonempty(candidate_rows, CANDIDATE_KEY, "candidate_outcomes.csv")
    candidate_by_key = {tuple(row[column] for column in CANDIDATE_KEY): row for row in candidate_rows}
    for row in candidate_rows:
        if row["cache_flag"].lower() != "false":
            raise ProtocolValidationError("candidate_outcomes.csv contains cache-derived row")
        if row["success"].lower() not in {"true", "false"}:
            raise ProtocolValidationError("candidate_outcomes.csv success must be boolean")
        numeric_values = [float(row[column]) for column in ("provider_generation_usd", "fallback_generation_usd", "model_api_cost_usd", "generation_latency_ms")]
        if not all(math.isfinite(value) and value >= 0 for value in numeric_values):
            raise ProtocolValidationError("candidate_outcomes.csv has negative cost or latency")
        total = float(row["provider_generation_usd"]) + float(row["fallback_generation_usd"])
        if abs(float(row["model_api_cost_usd"]) - total) > 0.000001:
            raise ProtocolValidationError("model_api_cost_usd does not equal named cost components")

    expected_candidate_keys = set()
    for benchmark, entry in protocol["benchmarks"].items():
        for task_id in entry["task_ids"]:
            for candidate in protocol["candidates"]:
                for replicate in range(entry["outcome_replicates_per_task_candidate"]):
                    expected_candidate_keys.add((benchmark, task_id, candidate, str(replicate)))
    if candidate_keys != expected_candidate_keys:
        raise ProtocolValidationError("candidate-outcome matrix does not match protocol coverage")

    route_rows = _read_csv(bundle_dir / "routes.csv")
    _require_columns(route_rows, set(ROUTE_KEY) | {
        "selected_candidate", "confidence", "fallback_path", "decision_latency_ms", "package_version",
        "router_service_usd", "configuration_digest", "route_vector_hash",
    }, "routes.csv")
    route_keys = _unique_nonempty(route_rows, ROUTE_KEY, "routes.csv")
    route_by_key = {tuple(row[column] for column in ROUTE_KEY): row for row in route_rows}
    expected_route_keys = set()
    router_ids = set(manifest.get("router_config_ids", []))
    if len(router_ids) != len(protocol["routers"]):
        raise ProtocolValidationError("manifest router_config_ids must cover every declared router")
    for benchmark, entry in protocol["benchmarks"].items():
        for task_id in entry["task_ids"]:
            for router_id in router_ids:
                for seed in range(entry["routing_seed_count"]):
                    expected_route_keys.add((router_id, benchmark, task_id, str(seed)))
    if route_keys != expected_route_keys:
        raise ProtocolValidationError("routes.csv does not match protocol coverage")
    if any(row["selected_candidate"] not in protocol["candidates"] for row in route_rows):
        raise ProtocolValidationError("routes.csv contains undeclared candidate")
    for row in route_rows:
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ProtocolValidationError("routes.csv confidence must lie in [0, 1]")
        route_values = [float(row[column]) for column in ("decision_latency_ms", "router_service_usd")]
        if not all(math.isfinite(value) and value >= 0 for value in route_values):
            raise ProtocolValidationError("routes.csv has negative cost or latency")

    outcome_rows = _read_csv(bundle_dir / "outcomes.csv")
    _require_columns(outcome_rows, set(ROUTE_KEY) | {"outcome_replicate", "candidate_outcome_key"}, "outcomes.csv")
    outcome_keys = _unique_nonempty(outcome_rows, ROUTE_KEY + ("outcome_replicate",), "outcomes.csv")
    expected_outcome_count = sum(entry["total_route_rows"] * entry["outcome_replicates_per_task_candidate"] for entry in protocol["benchmarks"].values())
    if len(outcome_keys) != expected_outcome_count:
        raise ProtocolValidationError("outcomes.csv has wrong joined outcome count")
    for row in outcome_rows:
        candidate_key = tuple(row["candidate_outcome_key"].split("|"))
        if candidate_key not in candidate_keys:
            raise ProtocolValidationError("outcomes.csv has unresolved candidate-outcome foreign key")
        route_key = tuple(row[column] for column in ROUTE_KEY)
        route = route_by_key[route_key]
        expected_key = (route["benchmark_id"], route["task_id"], route["selected_candidate"], row["outcome_replicate"])
        if candidate_key != expected_key:
            raise ProtocolValidationError("outcomes.csv does not join each route to its selected candidate outcome")

    result_rows = _read_csv(bundle_dir / "results.csv")
    _require_columns(result_rows, set(ROUTE_KEY) | {"outcome_replicate", "candidate_outcome_key"}, "results.csv")
    result_keys = _unique_nonempty(result_rows, ROUTE_KEY + ("outcome_replicate",), "results.csv")
    if result_keys != outcome_keys:
        raise ProtocolValidationError("results.csv does not have one row per locked joined outcome")
    outcome_by_key = {tuple(row[column] for column in ROUTE_KEY + ("outcome_replicate",)): row for row in outcome_rows}
    for row in result_rows:
        key = tuple(row[column] for column in ROUTE_KEY + ("outcome_replicate",))
        if row["candidate_outcome_key"] != outcome_by_key[key]["candidate_outcome_key"]:
            raise ProtocolValidationError("results.csv has a result-to-outcome lineage mismatch")

    trace_digests = {
        json.loads(line)["digest"] for line in (bundle_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line
    }
    if {row["raw_trace_digest"] for row in candidate_rows} - trace_digests:
        raise ProtocolValidationError("candidate outcome references missing raw trace digest")

    provenance = json.loads((bundle_dir / "provenance.json").read_text(encoding="utf-8"))
    excluded = provenance.get("excluded_historical_artifacts")
    if not isinstance(excluded, list) or not excluded or not all(
        isinstance(entry, dict) and set(entry) == {"path", "reason"} and entry["path"] and entry["reason"]
        for entry in excluded
    ):
        raise ProtocolValidationError("provenance must record each excluded historical artifact and reason")
