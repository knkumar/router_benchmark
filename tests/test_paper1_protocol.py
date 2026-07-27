from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from router_benchmark.protocol.protocol_tools import (
    ProtocolValidationError,
    load_yaml,
    validate_analysis_protocol,
    validate_cost_spec,
    validate_rebuild_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_has_valid_scope_and_count_arithmetic() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    validate_rebuild_protocol(protocol)


def test_protocol_accepts_the_authorized_execution_budget() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    validate_rebuild_protocol(protocol, require_approved_budget=True)


def test_protocol_refuses_approved_execution_with_placeholder_provenance() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    protocol["execution_budget"]["approval_status"] = "approved"
    protocol["model_snapshots"]["cheap-small"] = "UNCONFIRMED"
    with pytest.raises(ProtocolValidationError, match="model_snapshots"):
        validate_rebuild_protocol(protocol, require_approved_budget=True)


def test_protocol_has_concrete_execution_provenance_except_author_approval() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    protocol["execution_budget"]["approval_status"] = "approved"
    validate_rebuild_protocol(protocol, require_approved_budget=True)


def test_protocol_rejects_undeclared_router_and_bad_total() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    wrong_router = deepcopy(protocol)
    wrong_router["routers"][0] = "NVIDIA AI Blueprint LLM Router (live)"
    with pytest.raises(ProtocolValidationError, match="routers"):
        validate_rebuild_protocol(wrong_router)

    wrong_total = deepcopy(protocol)
    benchmark = next(iter(wrong_total["benchmarks"]))
    wrong_total["benchmarks"][benchmark]["total_outcome_rows"] -= 1
    with pytest.raises(ProtocolValidationError, match="total_outcome_rows"):
        validate_rebuild_protocol(wrong_total)


def test_protocol_rejects_missing_prespecified_router_pair() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    protocol["analysis_paired_comparisons"].pop()
    with pytest.raises(ProtocolValidationError, match="analysis_paired_comparisons"):
        validate_rebuild_protocol(protocol)


def test_protocol_rejects_baseline_seed_for_deterministic_policy() -> None:
    protocol = load_yaml(ROOT / "protocol/paper1_rebuild.yaml")
    protocol["baselines"][0]["routing_seeds"] = [1]
    with pytest.raises(ProtocolValidationError, match="must not declare routing seeds"):
        validate_rebuild_protocol(protocol)


def test_analysis_and_cost_contracts_validate() -> None:
    validate_analysis_protocol(load_yaml(ROOT / "protocol/analysis.yaml"))
    validate_cost_spec(load_yaml(ROOT / "protocol/cost_spec.yaml"))

