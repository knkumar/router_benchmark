from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from router_benchmark.protocol.protocol_tools import ProtocolValidationError, load_yaml
from router_benchmark.scripts.preflight_full_run import (
    candidate_reservations_from_full_protocol,
    validate_full_run_protocol,
)
from test_dry_run_preflight import _protocol


def _full_protocol() -> dict:
    protocol = deepcopy(_protocol())
    protocol["execution_budget"] = {
        "estimated_api_usd": "5",
        "estimated_infrastructure_usd": "0",
        "estimated_wall_time": "one minute",
        "stopping_rule": "fixture",
        "approval_status": "approved",
    }
    compact = {}
    for benchmark in protocol["benchmarks"]:
        for candidate in protocol["candidates"]:
            compact[f"{benchmark}|{candidate}"] = 0.2
    compact["tau2-bench (live)|mid-general"] = 1.0
    compact["tau2-bench (live)|strong-frontier"] = 1.6
    compact["WebArena (live)|strong-frontier"] = 0.4
    protocol["full_run_cost_reservations"] = {
        "total_cap_usd": 20,
        "pricing_basis": "fixture full-run reservations",
        "candidate_model_api_usd_by_benchmark_candidate": compact,
        "router_service_usd": {"total_reserved_usd": 1.0},
        "noncandidate_model_api_usd": {"total_reserved_usd": 1.0},
        "request_limits": {
            "RouterBench (live)": {"max_provider_calls_per_candidate_cell": 0, "max_output_tokens_per_call": 0, "max_steps": 0},
            "BFCL v4 (live)": {"max_provider_calls_per_candidate_cell": 1, "max_output_tokens_per_call": 256, "max_steps": 1},
            "tau2-bench (live)": {"max_provider_calls_per_candidate_cell": 60, "max_output_tokens_per_call": 1024, "max_steps": 30},
            "WebArena (live)": {"max_provider_calls_per_candidate_cell": 30, "max_output_tokens_per_call": 384, "max_steps": 30},
        },
    }
    return protocol


def test_full_run_preflight_accepts_compact_reservations() -> None:
    protocol = _full_protocol()

    validate_full_run_protocol(protocol)

    reservations = candidate_reservations_from_full_protocol(protocol)
    assert len(reservations) == 24
    assert reservations["tau2-bench (live)|2-task|strong-frontier|0"] == 1.6


def test_full_run_preflight_rejects_pending_author_approval() -> None:
    protocol = _full_protocol()
    protocol["execution_budget"]["approval_status"] = "pending_author_approval"
    with pytest.raises(ProtocolValidationError, match="approval_status"):
        validate_full_run_protocol(protocol)


def test_full_run_preflight_rejects_missing_compact_reservation() -> None:
    protocol = _full_protocol()
    protocol["full_run_cost_reservations"]["candidate_model_api_usd_by_benchmark_candidate"].pop(
        "RouterBench (live)|cheap-small"
    )
    with pytest.raises(ProtocolValidationError, match="cover every benchmark/candidate"):
        validate_full_run_protocol(protocol)


def test_current_rebuild_protocol_is_approved_for_full_execution() -> None:
    protocol = load_yaml(Path("protocol/paper1_rebuild.yaml"))
    validate_full_run_protocol(protocol)
