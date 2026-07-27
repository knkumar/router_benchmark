from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from router_benchmark.protocol.protocol_tools import ProtocolValidationError
from router_benchmark.scripts.preflight_dry_run import validate_dry_run_protocol


def _entry(task_id: str) -> dict[str, object]:
    return {
        "subset_id": "fixture", "task_ids": [task_id, f"{task_id}-other"],
        "task_id_sha256": hashlib.sha256(f"{task_id}\n{task_id}-other".encode()).hexdigest(),
        "router_trials_per_task": 1, "routing_seed_count": 1,
        "outcome_replicates_per_task_candidate": 1, "total_route_rows": 8,
        "total_outcome_rows": 6,
    }


def _protocol() -> dict[str, object]:
    routers = ["LiteLLM Router (live)", "Aurelio Semantic Router (live)", "RouteLLM (live)", "vLLM Semantic Router (live)"]
    benchmarks = ["RouterBench (live)", "BFCL v4 (live)", "tau2-bench (live)", "WebArena (live)"]
    return {
        "routers": routers, "candidates": ["cheap-small", "mid-general", "strong-frontier"],
        "calibration_policy": "out_of_the_box_only", "cross_benchmark_decision": "weighted_success_reported_separately_from_cost",
        "baselines": [
            {"name": "Always-Cheapest Baseline (live)", "candidate_policy": "always cheap-small", "randomization": "deterministic", "routing_seeds": [], "cascade_order": [], "stopping_rule": "fixture", "fallback_behavior": "none"},
            {"name": "Always-Strongest Baseline (live)", "candidate_policy": "always strong-frontier", "randomization": "deterministic", "routing_seeds": [], "cascade_order": [], "stopping_rule": "fixture", "fallback_behavior": "none"},
        ],
        "analysis_paired_comparisons": [[a, b] for index, a in enumerate(routers) for b in routers[index + 1:]],
        "pricing": {"as_of": "2026-07-20"},
        "model_snapshots": {candidate: "fixture-2026-07-20" for candidate in ["cheap-small", "mid-general", "strong-frontier"]},
        "grader_versions": {benchmark: "fixture-2026-07-20" for benchmark in benchmarks},
        "benchmarks": {benchmark: _entry(f"{index}-task") for index, benchmark in enumerate(benchmarks)},
        "execution_budget": {"estimated_api_usd": "9", "estimated_infrastructure_usd": "0", "estimated_wall_time": "one minute", "stopping_rule": "fixture", "approval_status": "approved"},
    }


def _dry_from(frozen: dict[str, object]) -> dict[str, object]:
    dry = deepcopy(frozen)
    for entry in dry["benchmarks"].values():
        task_id = entry["task_ids"][0]
        entry["task_ids"] = [task_id]
        entry["task_id_sha256"] = hashlib.sha256(task_id.encode()).hexdigest()
        entry["total_route_rows"] = 4
        entry["total_outcome_rows"] = 3
    dry["diagnostic_only"] = True
    dry["dry_run_budget_cap_usd"] = 10
    candidate_reservations = {}
    for benchmark, entry in dry["benchmarks"].items():
        for task_id in entry["task_ids"]:
            for candidate in dry["candidates"]:
                reservation = 0.2
                if benchmark == "tau2-bench (live)" and candidate == "mid-general":
                    reservation = 1.0
                if benchmark == "tau2-bench (live)" and candidate == "strong-frontier":
                    reservation = 1.6
                if benchmark == "WebArena (live)" and candidate == "strong-frontier":
                    reservation = 0.4
                candidate_reservations[f"{benchmark}|{task_id}|{candidate}|0"] = reservation
    dry["dry_run_cost_reservations"] = {
        "total_cap_usd": 10,
        "pricing_basis": "fixture upper-bound reservations",
        "candidate_model_api_usd": candidate_reservations,
        "router_service_usd": {"total_reserved_usd": 1.0},
        "noncandidate_model_api_usd": {"total_reserved_usd": 1.0},
        "request_limits": {
            "RouterBench (live)": {"max_provider_calls_per_candidate_cell": 0, "max_output_tokens_per_call": 0, "max_steps": 0},
            "BFCL v4 (live)": {"max_provider_calls_per_candidate_cell": 1, "max_output_tokens_per_call": 256, "max_steps": 1},
            "tau2-bench (live)": {"max_provider_calls_per_candidate_cell": 60, "max_output_tokens_per_call": 1024, "max_steps": 30},
            "WebArena (live)": {"max_provider_calls_per_candidate_cell": 30, "max_output_tokens_per_call": 384, "max_steps": 30},
        },
    }
    return dry


def test_dry_run_preflight_accepts_bounded_frozen_subset() -> None:
    frozen = _protocol()
    validate_dry_run_protocol(_dry_from(frozen), frozen)


def test_dry_run_preflight_rejects_cap_over_ten() -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    dry["dry_run_budget_cap_usd"] = 10.01
    dry["dry_run_cost_reservations"]["total_cap_usd"] = 10.01
    with pytest.raises(ProtocolValidationError, match="budget_cap"):
        validate_dry_run_protocol(dry, frozen)


def test_dry_run_preflight_rejects_missing_cell_reservation() -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    dry["dry_run_cost_reservations"]["candidate_model_api_usd"].pop(next(iter(dry["dry_run_cost_reservations"]["candidate_model_api_usd"])))
    with pytest.raises(ProtocolValidationError, match="enumerate every dry-run cell"):
        validate_dry_run_protocol(dry, frozen)


def test_dry_run_preflight_rejects_reservations_over_cap() -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    for key in list(dry["dry_run_cost_reservations"]["candidate_model_api_usd"]):
        dry["dry_run_cost_reservations"]["candidate_model_api_usd"][key] = 2.0
    with pytest.raises(ProtocolValidationError, match="reservations exceed"):
        validate_dry_run_protocol(dry, frozen)


def test_dry_run_preflight_rejects_reservation_below_output_floor() -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    key = "tau2-bench (live)|2-task|strong-frontier|0"
    dry["dry_run_cost_reservations"]["candidate_model_api_usd"][key] = 1.0
    dry["dry_run_cost_reservations"]["router_service_usd"]["total_reserved_usd"] = 0.0
    dry["dry_run_cost_reservations"]["noncandidate_model_api_usd"]["total_reserved_usd"] = 0.0
    with pytest.raises(ProtocolValidationError, match="output-token floor"):
        validate_dry_run_protocol(dry, frozen)


def test_dry_run_preflight_rejects_incomplete_request_limits() -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    dry["dry_run_cost_reservations"]["request_limits"]["tau2-bench (live)"].pop("max_steps")
    with pytest.raises(ProtocolValidationError, match="request_limits"):
        validate_dry_run_protocol(dry, frozen)
