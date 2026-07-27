"""Validation for the prespecified Paper 1 rebuild contract.

The validator intentionally has no provider-call path.  It validates study
scope before an execution runner is allowed to create a canonical bundle.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml


class ProtocolValidationError(ValueError):
    """Raised when a rebuild contract is incomplete or internally invalid."""


REQUIRED_ROUTERS = {
    "LiteLLM Router (live)",
    "Aurelio Semantic Router (live)",
    "RouteLLM (live)",
    "vLLM Semantic Router (live)",
}
REQUIRED_BENCHMARKS = {
    "RouterBench (live)",
    "BFCL v4 (live)",
    "tau2-bench (live)",
    "WebArena (live)",
}
REQUIRED_CANDIDATES = {"cheap-small", "mid-general", "strong-frontier"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{path} must contain a mapping")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{label} must be a mapping")
    return value


def _require_exact_set(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, list) or set(value) != expected or len(value) != len(expected):
        raise ProtocolValidationError(f"{label} must declare exactly {sorted(expected)}")


def _sha256_ids(task_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


def _is_execution_placeholder(value: Any) -> bool:
    """Return whether an execution field still contains a planning placeholder."""
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower()
    return normalized == "unconfirmed" or "must be recorded before run" in normalized or "digest required" in normalized


def validate_rebuild_protocol(protocol: dict[str, Any], *, require_approved_budget: bool = False) -> None:
    """Validate study scope and count arithmetic without inspecting outcomes."""
    _require_exact_set(protocol.get("routers"), REQUIRED_ROUTERS, "routers")
    _require_exact_set(protocol.get("candidates"), REQUIRED_CANDIDATES, "candidates")
    expected_pairs = {tuple(pair) for pair in combinations(sorted(REQUIRED_ROUTERS), 2)}
    comparisons = protocol.get("analysis_paired_comparisons")
    if not isinstance(comparisons, list) or {tuple(sorted(pair)) for pair in comparisons if isinstance(pair, list) and len(pair) == 2} != expected_pairs:
        raise ProtocolValidationError("analysis_paired_comparisons must declare every router pair exactly once")
    if protocol.get("calibration_policy") != "out_of_the_box_only":
        raise ProtocolValidationError("calibration_policy must be out_of_the_box_only for this rebuild")
    if protocol.get("cross_benchmark_decision") != "weighted_success_reported_separately_from_cost":
        raise ProtocolValidationError("cross_benchmark_decision must not claim production traffic or utility")
    baselines = protocol.get("baselines")
    required_baselines = {"Always-Cheapest Baseline (live)", "Always-Strongest Baseline (live)"}
    if not isinstance(baselines, list) or {entry.get("name") for entry in baselines if isinstance(entry, dict)} != required_baselines:
        raise ProtocolValidationError("baselines must declare the paired cheapest and strongest policies")
    for baseline in baselines:
        required = {"name", "candidate_policy", "randomization", "routing_seeds", "cascade_order", "stopping_rule", "fallback_behavior"}
        if set(baseline) != required:
            raise ProtocolValidationError(f"baseline {baseline.get('name', '<unknown>')} has missing or undeclared fields")
        if baseline["randomization"] != "deterministic":
            raise ProtocolValidationError("declared static baselines must be deterministic")
        if baseline["routing_seeds"] != []:
            raise ProtocolValidationError("deterministic baselines must not declare routing seeds")

    pricing = _require_mapping(protocol.get("pricing"), "pricing")
    if not isinstance(pricing.get("as_of"), str) or not pricing["as_of"]:
        raise ProtocolValidationError("pricing.as_of is required")
    snapshots = _require_mapping(protocol.get("model_snapshots"), "model_snapshots")
    if set(snapshots) != REQUIRED_CANDIDATES:
        raise ProtocolValidationError("model_snapshots must cover every candidate tier")

    benchmarks = _require_mapping(protocol.get("benchmarks"), "benchmarks")
    if set(benchmarks) != REQUIRED_BENCHMARKS:
        raise ProtocolValidationError("benchmarks must declare exactly the four study benchmarks")
    grader_versions = _require_mapping(protocol.get("grader_versions"), "grader_versions")
    if set(grader_versions) != REQUIRED_BENCHMARKS:
        raise ProtocolValidationError("grader_versions must cover every benchmark")

    for benchmark, entry in benchmarks.items():
        entry = _require_mapping(entry, f"benchmarks.{benchmark}")
        task_ids = entry.get("task_ids")
        if not isinstance(task_ids, list) or not task_ids or not all(isinstance(task, str) and task for task in task_ids):
            raise ProtocolValidationError(f"benchmarks.{benchmark}.task_ids must be nonempty strings")
        if len(task_ids) != len(set(task_ids)):
            raise ProtocolValidationError(f"benchmarks.{benchmark}.task_ids contains duplicates")
        if entry.get("task_id_sha256") != _sha256_ids(task_ids):
            raise ProtocolValidationError(f"benchmarks.{benchmark}.task_id_sha256 does not match task_ids")
        for field in ("router_trials_per_task", "routing_seed_count", "outcome_replicates_per_task_candidate"):
            if not isinstance(entry.get(field), int) or entry[field] < 1:
                raise ProtocolValidationError(f"benchmarks.{benchmark}.{field} must be a positive integer")
        expected_routes = len(task_ids) * entry["router_trials_per_task"] * len(REQUIRED_ROUTERS)
        expected_outcomes = len(task_ids) * entry["outcome_replicates_per_task_candidate"] * len(REQUIRED_CANDIDATES)
        if entry.get("total_route_rows") != expected_routes:
            raise ProtocolValidationError(f"benchmarks.{benchmark}.total_route_rows must equal {expected_routes}")
        if entry.get("total_outcome_rows") != expected_outcomes:
            raise ProtocolValidationError(f"benchmarks.{benchmark}.total_outcome_rows must equal {expected_outcomes}")
        if not isinstance(entry.get("subset_id"), str) or not entry["subset_id"]:
            raise ProtocolValidationError(f"benchmarks.{benchmark}.subset_id is required")

    budget = _require_mapping(protocol.get("execution_budget"), "execution_budget")
    required_budget_fields = {"estimated_api_usd", "estimated_infrastructure_usd", "estimated_wall_time", "stopping_rule", "approval_status"}
    if set(budget) != required_budget_fields:
        raise ProtocolValidationError("execution_budget has missing or undeclared fields")
    if require_approved_budget and budget["approval_status"] != "approved":
        raise ProtocolValidationError("execution_budget.approval_status must be approved before execution")
    if require_approved_budget:
        unresolved_models = [candidate for candidate, snapshot in snapshots.items() if _is_execution_placeholder(snapshot)]
        if unresolved_models:
            raise ProtocolValidationError(f"model_snapshots must be concrete before execution: {sorted(unresolved_models)}")
        unresolved_graders = [benchmark for benchmark, version in grader_versions.items() if _is_execution_placeholder(version)]
        if unresolved_graders:
            raise ProtocolValidationError(f"grader_versions must be concrete before execution: {sorted(unresolved_graders)}")
        unresolved_budget = [
            field for field in ("estimated_api_usd", "estimated_infrastructure_usd", "estimated_wall_time", "stopping_rule")
            if _is_execution_placeholder(budget[field])
        ]
        if unresolved_budget:
            raise ProtocolValidationError(f"execution_budget has unresolved execution fields: {sorted(unresolved_budget)}")


def validate_analysis_protocol(analysis: dict[str, Any]) -> None:
    required = {
        "estimands", "independent_unit", "paired_comparisons", "confidence_level",
        "alpha", "multiplicity", "benchmark_weighting", "resampling", "missing_data",
    }
    if set(analysis) != required:
        raise ProtocolValidationError("analysis protocol has missing or undeclared fields")
    if analysis["independent_unit"] != "benchmark_id x task_id":
        raise ProtocolValidationError("independent_unit must be benchmark_id x task_id")
    if not isinstance(analysis["paired_comparisons"], list) or not analysis["paired_comparisons"]:
        raise ProtocolValidationError("paired_comparisons must be nonempty")


def validate_cost_spec(cost_spec: dict[str, Any]) -> None:
    components = cost_spec.get("components")
    if not isinstance(components, list) or not components:
        raise ProtocolValidationError("cost_spec.components must be nonempty")
    names = set()
    for component in components:
        component = _require_mapping(component, "cost component")
        for field in ("name", "measurement", "unit", "rate", "included"):
            if field not in component:
                raise ProtocolValidationError(f"cost component missing {field}")
        if component["name"] in names:
            raise ProtocolValidationError(f"duplicate cost component {component['name']}")
        names.add(component["name"])
    if cost_spec.get("reported_metric") != "model API cost":
        raise ProtocolValidationError("reported_metric must be model API cost until all deployment costs are measured")
