"""Validate the full Paper 1 rebuild before any provider call.

The full-run contract executes the frozen study scope, not a diagnostic
subset.  This module performs only local validation and cost-reservation
expansion; it never constructs benchmark adapters or provider clients.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from router_benchmark.protocol.candidate_runner import CandidateCell, cell_key_string
from router_benchmark.protocol.protocol_tools import (
    ProtocolValidationError,
    load_yaml,
    validate_rebuild_protocol,
)
from router_benchmark.scripts.preflight_dry_run import OUTPUT_USD_PER_TOKEN


RESERVATION_FIELD = "full_run_cost_reservations"


def full_run_candidate_cells(protocol: Mapping[str, Any]) -> list[CandidateCell]:
    return [
        CandidateCell(benchmark, task_id, candidate, replicate)
        for benchmark, entry in protocol["benchmarks"].items()
        for task_id in entry["task_ids"]
        for candidate in protocol["candidates"]
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


def _reservation_mapping(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    reservations = protocol.get(RESERVATION_FIELD)
    if not isinstance(reservations, Mapping):
        raise ProtocolValidationError(f"{RESERVATION_FIELD} is required")
    return reservations


def candidate_reservations_from_full_protocol(protocol: Mapping[str, Any]) -> dict[str, float]:
    """Return exact per-cell candidate reservations for CandidateStageRunner.

    To keep the full protocol reviewable, it may either enumerate every cell
    under ``candidate_model_api_usd`` or provide compact values under
    ``candidate_model_api_usd_by_benchmark_candidate`` using keys of the form
    ``"<benchmark>|<candidate>"``.  The compact form is expanded over every
    task ID and replicate in the frozen protocol.
    """
    reservations = _reservation_mapping(protocol)
    exact = reservations.get("candidate_model_api_usd")
    compact = reservations.get("candidate_model_api_usd_by_benchmark_candidate")
    if exact is not None and compact is not None:
        raise ProtocolValidationError(
            f"{RESERVATION_FIELD} must not declare both exact and compact candidate reservations"
        )
    expected = {cell_key_string(cell) for cell in full_run_candidate_cells(protocol)}
    if exact is not None:
        if not isinstance(exact, Mapping):
            raise ProtocolValidationError(f"{RESERVATION_FIELD}.candidate_model_api_usd must be a mapping")
        if set(exact) != expected:
            missing = sorted(expected - set(exact))
            extra = sorted(set(exact) - expected)
            raise ProtocolValidationError(
                f"{RESERVATION_FIELD}.candidate_model_api_usd must enumerate every full-run cell; "
                f"missing={missing}; extra={extra}"
            )
        return {
            key: _finite_nonnegative(value, f"{RESERVATION_FIELD}.candidate_model_api_usd[{key}]")
            for key, value in exact.items()
        }
    if not isinstance(compact, Mapping):
        raise ProtocolValidationError(
            f"{RESERVATION_FIELD}.candidate_model_api_usd_by_benchmark_candidate must be a mapping"
        )
    expected_compact = {
        f"{benchmark}|{candidate}"
        for benchmark in protocol["benchmarks"]
        for candidate in protocol["candidates"]
    }
    if set(compact) != expected_compact:
        missing = sorted(expected_compact - set(compact))
        extra = sorted(set(compact) - expected_compact)
        raise ProtocolValidationError(
            f"{RESERVATION_FIELD}.candidate_model_api_usd_by_benchmark_candidate must cover every benchmark/candidate; "
            f"missing={missing}; extra={extra}"
        )
    parsed_compact = {
        key: _finite_nonnegative(
            value,
            f"{RESERVATION_FIELD}.candidate_model_api_usd_by_benchmark_candidate[{key}]",
        )
        for key, value in compact.items()
    }
    return {
        cell_key_string(cell): parsed_compact[f"{cell.benchmark_id}|{cell.candidate_id}"]
        for cell in full_run_candidate_cells(protocol)
    }


def router_service_reservation_from_full_protocol(protocol: Mapping[str, Any]) -> float:
    reservations = _reservation_mapping(protocol)
    router_service = reservations.get("router_service_usd")
    if not isinstance(router_service, Mapping):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.router_service_usd must be a mapping")
    return _finite_nonnegative(
        router_service.get("total_reserved_usd"),
        f"{RESERVATION_FIELD}.router_service_usd.total_reserved_usd",
    )


def noncandidate_model_reservation_from_full_protocol(protocol: Mapping[str, Any]) -> float:
    reservations = _reservation_mapping(protocol)
    noncandidate = reservations.get("noncandidate_model_api_usd")
    if not isinstance(noncandidate, Mapping):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.noncandidate_model_api_usd must be a mapping")
    return _finite_nonnegative(
        noncandidate.get("total_reserved_usd"),
        f"{RESERVATION_FIELD}.noncandidate_model_api_usd.total_reserved_usd",
    )


def external_metered_reservation_from_full_protocol(protocol: Mapping[str, Any]) -> float:
    return (
        router_service_reservation_from_full_protocol(protocol)
        + noncandidate_model_reservation_from_full_protocol(protocol)
    )


def request_limits_from_full_protocol(protocol: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    reservations = _reservation_mapping(protocol)
    request_limits = reservations.get("request_limits")
    if not isinstance(request_limits, Mapping) or set(request_limits) != set(protocol["benchmarks"]):
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits must cover every full-run benchmark")
    parsed: dict[str, dict[str, int]] = {}
    required = {"max_provider_calls_per_candidate_cell", "max_output_tokens_per_call", "max_steps"}
    for benchmark, limits in request_limits.items():
        if not isinstance(limits, Mapping):
            raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits[{benchmark}] must be a mapping")
        if set(limits) != required:
            raise ProtocolValidationError(f"{RESERVATION_FIELD}.request_limits[{benchmark}] has missing or undeclared fields")
        parsed[benchmark] = {}
        for field in required:
            value = limits[field]
            if not isinstance(value, int) or value < 0:
                raise ProtocolValidationError(
                    f"{RESERVATION_FIELD}.request_limits[{benchmark}].{field} must be a nonnegative integer"
                )
            parsed[benchmark][field] = value
        if parsed[benchmark]["max_provider_calls_per_candidate_cell"] > 0 and parsed[benchmark]["max_output_tokens_per_call"] == 0:
            raise ProtocolValidationError(
                f"{RESERVATION_FIELD}.request_limits[{benchmark}].max_output_tokens_per_call must be positive when provider calls are allowed"
            )
    return parsed


def validate_candidate_reservation_floors(protocol: Mapping[str, Any]) -> None:
    reservations = candidate_reservations_from_full_protocol(protocol)
    request_limits = request_limits_from_full_protocol(protocol)
    for cell in full_run_candidate_cells(protocol):
        key = cell_key_string(cell)
        limits = request_limits[cell.benchmark_id]
        output_floor = (
            limits["max_provider_calls_per_candidate_cell"]
            * limits["max_output_tokens_per_call"]
            * OUTPUT_USD_PER_TOKEN[cell.candidate_id]
        )
        if reservations[key] + 1e-12 < output_floor:
            raise ProtocolValidationError(
                f"{RESERVATION_FIELD}.candidate reservation for {key} is below declared output-token floor"
            )


def validate_full_run_protocol(protocol: dict[str, Any]) -> None:
    """Reject a full execution plan that lacks approval, budget, or limits."""
    validate_rebuild_protocol(protocol, require_approved_budget=True)
    if protocol.get("diagnostic_only") is True:
        raise ProtocolValidationError("full run must not be marked diagnostic_only")
    reservations = _reservation_mapping(protocol)
    total_cap = _finite_nonnegative(reservations.get("total_cap_usd"), f"{RESERVATION_FIELD}.total_cap_usd")
    if total_cap <= 0:
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.total_cap_usd must be positive")
    if not isinstance(reservations.get("pricing_basis"), str) or not reservations["pricing_basis"].strip():
        raise ProtocolValidationError(f"{RESERVATION_FIELD}.pricing_basis is required")
    request_limits_from_full_protocol(protocol)
    candidate_reserved = candidate_reservations_from_full_protocol(protocol)
    validate_candidate_reservation_floors(protocol)
    external_reserved = external_metered_reservation_from_full_protocol(protocol)
    if sum(candidate_reserved.values()) + external_reserved > total_cap + 1e-9:
        raise ProtocolValidationError("full-run candidate and external metered reservations exceed the budget cap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_yaml(args.protocol)
    validate_full_run_protocol(protocol)
    reservations = candidate_reservations_from_full_protocol(protocol)
    rows = len(full_run_candidate_cells(protocol))
    cap = float(protocol[RESERVATION_FIELD]["total_cap_usd"])
    print(
        f"Full-run preflight passed: cap=${cap:.2f}; "
        f"{rows} candidate cells; ${sum(reservations.values()):.2f} candidate reserve."
    )


if __name__ == "__main__":
    main()
