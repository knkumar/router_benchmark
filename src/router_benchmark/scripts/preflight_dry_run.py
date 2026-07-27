"""Validate a bounded diagnostic dry-run before any provider call.

The dry-run contract must use a strict subset of the frozen study task IDs,
cover all four benchmarks and candidate tiers, and state a hard total cap of
at most ten USD.  This command deliberately performs no network or provider
operations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from router_benchmark.protocol.candidate_runner import CandidateCell, cell_key_string
from router_benchmark.protocol.protocol_tools import (
    ProtocolValidationError,
    load_yaml,
    validate_rebuild_protocol,
)


MAX_DRY_RUN_CAP_USD = 10.0
RESERVATION_FIELD = "dry_run_cost_reservations"
OUTPUT_USD_PER_TOKEN = {
    "cheap-small": 1.25 / 1_000_000,
    "mid-general": 15.00 / 1_000_000,
    "strong-frontier": 25.00 / 1_000_000,
}


def dry_run_candidate_cells(dry: dict[str, Any]) -> list[CandidateCell]:
    return [
        CandidateCell(benchmark, task_id, candidate, replicate)
        for benchmark, entry in dry["benchmarks"].items()
        for task_id in entry["task_ids"]
        for candidate in dry["candidates"]
        for replicate in range(entry["outcome_replicates_per_task_candidate"])
    ]


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ProtocolValidationError(f"{label} must be finite and nonnegative")
    return parsed


def candidate_reservations_from_protocol(dry: dict[str, Any]) -> dict[str, float]:
    reservations = dry.get(RESERVATION_FIELD)
    if not isinstance(reservations, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD} is required")
    candidate_reservations = reservations.get("candidate_model_api_usd")
    if not isinstance(candidate_reservations, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.candidate_model_api_usd must be a mapping")
    expected = {cell_key_string(cell) for cell in dry_run_candidate_cells(dry)}
    actual = set(candidate_reservations)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolValidationError(
            f"{RESERVATION_FIELD}.candidate_model_api_usd must enumerate every dry-run cell; "
            f"missing={missing}; extra={extra}"
        )
    return {
        key: _finite_nonnegative(value, f"{RESERVATION_FIELD}.candidate_model_api_usd[{key}]")
        for key, value in candidate_reservations.items()
    }


def router_service_reservation_from_protocol(dry: dict[str, Any]) -> float:
    reservations = dry.get(RESERVATION_FIELD)
    if not isinstance(reservations, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD} is required")
    router_service = reservations.get("router_service_usd")
    if not isinstance(router_service, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.router_service_usd must be a mapping")
    return _finite_nonnegative(
        router_service.get("total_reserved_usd"),
        f"{RESERVATION_FIELD}.router_service_usd.total_reserved_usd",
    )


def noncandidate_model_reservation_from_protocol(dry: dict[str, Any]) -> float:
    reservations = dry.get(RESERVATION_FIELD)
    if not isinstance(reservations, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD} is required")
    noncandidate = reservations.get("noncandidate_model_api_usd")
    if not isinstance(noncandidate, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.noncandidate_model_api_usd must be a mapping")
    return _finite_nonnegative(
        noncandidate.get("total_reserved_usd"),
        f"{RESERVATION_FIELD}.noncandidate_model_api_usd.total_reserved_usd",
    )


def external_metered_reservation_from_protocol(dry: dict[str, Any]) -> float:
    return router_service_reservation_from_protocol(dry) + noncandidate_model_reservation_from_protocol(dry)


def request_limits_from_protocol(dry: dict[str, Any]) -> dict[str, dict[str, int]]:
    reservations = dry.get(RESERVATION_FIELD)
    if not isinstance(reservations, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD} is required")
    request_limits = reservations.get("request_limits")
    if not isinstance(request_limits, dict) or set(request_limits) != set(dry["benchmarks"]):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits must cover every dry-run benchmark")
    parsed: dict[str, dict[str, int]] = {}
    for benchmark, limits in request_limits.items():
        if not isinstance(limits, dict):
            raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits[{benchmark}] must be a mapping")
        required = {"max_provider_calls_per_candidate_cell", "max_output_tokens_per_call", "max_steps"}
        if set(limits) != required:
            raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits[{benchmark}] has missing or undeclared fields")
        parsed[benchmark] = {}
        for field in required:
            value = limits[field]
            if not isinstance(value, int) or value < 0:
                raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits[{benchmark}].{field} must be a nonnegative integer")
            parsed[benchmark][field] = value
        if parsed[benchmark]["max_provider_calls_per_candidate_cell"] > 0 and parsed[benchmark]["max_output_tokens_per_call"] == 0:
            raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits[{benchmark}].max_output_tokens_per_call must be positive when provider calls are allowed")
    return parsed


def validate_candidate_reservation_floors(dry: dict[str, Any]) -> None:
    reservations = candidate_reservations_from_protocol(dry)
    request_limits = request_limits_from_protocol(dry)
    for cell in dry_run_candidate_cells(dry):
        key = cell_key_string(cell)
        limits = request_limits[cell.benchmark_id]
        output_floor = (
            limits["max_provider_calls_per_candidate_cell"]
            * limits["max_output_tokens_per_call"]
            * OUTPUT_USD_PER_TOKEN[cell.candidate_id]
        )
        if reservations[key] + 1e-12 < output_floor:
            raise ProtocolValidationError(
                f"{RESERVATION_FIELD}.candidate_model_api_usd[{key}] is below declared output-token floor"
            )


def validate_dry_run_protocol(dry: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Reject an execution plan that expands, mixes, or exceeds the dry-run scope."""
    validate_rebuild_protocol(dry, require_approved_budget=True)
    if dry.get("diagnostic_only") is not True:
        raise ProtocolValidationError("dry run must set diagnostic_only: true")
    try:
        cap = float(dry.get("dry_run_budget_cap_usd"))
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError("dry_run_budget_cap_usd must be numeric") from exc
    if not 0 < cap <= MAX_DRY_RUN_CAP_USD:
        raise ProtocolValidationError(f"dry_run_budget_cap_usd must be in (0, {MAX_DRY_RUN_CAP_USD}]")
    if dry.get("candidates") != frozen.get("candidates"):
        raise ProtocolValidationError("dry run must execute every frozen candidate tier")
    for benchmark, dry_entry in dry["benchmarks"].items():
        frozen_ids = set(frozen["benchmarks"][benchmark]["task_ids"])
        dry_ids = dry_entry["task_ids"]
        if not set(dry_ids) <= frozen_ids:
            raise ProtocolValidationError(f"dry run contains task IDs outside frozen scope for {benchmark}")
        if len(dry_ids) >= len(frozen_ids):
            raise ProtocolValidationError(f"dry run must be a strict task subset for {benchmark}")
    reservations = dry.get(RESERVATION_FIELD)
    if not isinstance(reservations, dict):
        raise ProtocolValidationError(f"{RESERVATION_FIELD} is required")
    total_cap = _finite_nonnegative(reservations.get("total_cap_usd"), f"{RESERVATION_FIELD}.total_cap_usd")
    if abs(total_cap - cap) > 1e-9:
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.total_cap_usd must equal dry_run_budget_cap_usd")
    if not isinstance(reservations.get("pricing_basis"), str) or not reservations["pricing_basis"].strip():
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.pricing_basis is required")
    request_limits_from_protocol(dry)
    candidate_reserved = candidate_reservations_from_protocol(dry)
    validate_candidate_reservation_floors(dry)
    external_metered_reserved = external_metered_reservation_from_protocol(dry)
    if sum(candidate_reserved.values()) + external_metered_reserved > cap + 1e-9:
        raise ProtocolValidationError("dry-run candidate and external metered reservations exceed the budget cap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-protocol", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    args = parser.parse_args()
    dry = load_yaml(args.dry_protocol)
    frozen = load_yaml(args.frozen_protocol)
    validate_rebuild_protocol(frozen)
    validate_dry_run_protocol(dry, frozen)
    print(f"Dry-run preflight passed: cap=${float(dry['dry_run_budget_cap_usd']):.2f}; diagnostic output only.")


if __name__ == "__main__":
    main()
