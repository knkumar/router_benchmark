#!/usr/bin/env python3
"""Report all local blockers for an approved full Paper 1 rebuild."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from router_benchmark.protocol.protocol_tools import (
    ProtocolValidationError,
    load_yaml,
    validate_rebuild_protocol,
)
from router_benchmark.scripts.preflight_full_run import (
    RESERVATION_FIELD,
    candidate_reservations_from_full_protocol,
    external_metered_reservation_from_full_protocol,
    full_run_candidate_cells,
    request_limits_from_full_protocol,
    validate_candidate_reservation_floors,
)


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower()
    return normalized == "unconfirmed" or "must be recorded before run" in normalized or "digest required" in normalized


def _check(blockers: list[str], name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except (ProtocolValidationError, RuntimeError, ValueError, FileNotFoundError) as exc:
        blockers.append(f"{name}: {exc}")


def _budget_blockers(protocol: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    budget = protocol.get("execution_budget")
    if not isinstance(budget, dict):
        return ["execution_budget: missing or not a mapping"]
    if budget.get("approval_status") != "approved":
        blockers.append("execution_budget.approval_status: must be approved before full execution")
    unresolved = [
        field for field in ("estimated_api_usd", "estimated_infrastructure_usd", "estimated_wall_time", "stopping_rule")
        if _is_placeholder(budget.get(field))
    ]
    if unresolved:
        blockers.append(f"execution_budget unresolved fields: {sorted(unresolved)}")
    return blockers


def _execution_identity_blockers(protocol: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    snapshots = protocol.get("model_snapshots")
    if isinstance(snapshots, dict):
        unresolved_models = [name for name, value in snapshots.items() if _is_placeholder(value)]
        if unresolved_models:
            blockers.append(f"model_snapshots unresolved fields: {sorted(unresolved_models)}")
    else:
        blockers.append("model_snapshots: missing or not a mapping")
    graders = protocol.get("grader_versions")
    if isinstance(graders, dict):
        unresolved_graders = [name for name, value in graders.items() if _is_placeholder(value)]
        if unresolved_graders:
            blockers.append(f"grader_versions unresolved fields: {sorted(unresolved_graders)}")
    else:
        blockers.append("grader_versions: missing or not a mapping")
    return blockers


def _reservation_blockers(protocol: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    reservations = protocol.get(RESERVATION_FIELD)
    if not isinstance(reservations, dict):
        return [f"{RESERVATION_FIELD}: missing or not a mapping"]
    if not reservations.get("pricing_basis"):
        blockers.append(f"{RESERVATION_FIELD}.pricing_basis: required")
    if float(reservations.get("total_cap_usd", 0) or 0) <= 0:
        blockers.append(f"{RESERVATION_FIELD}.total_cap_usd: must be positive")
    _check(blockers, "request_limits", lambda: request_limits_from_full_protocol(protocol))
    _check(blockers, "candidate reservations", lambda: candidate_reservations_from_full_protocol(protocol))
    _check(blockers, "reservation floors", lambda: validate_candidate_reservation_floors(protocol))
    _check(blockers, "external metered reservations", lambda: external_metered_reservation_from_full_protocol(protocol))
    try:
        candidate_total = sum(candidate_reservations_from_full_protocol(protocol).values())
        external_total = external_metered_reservation_from_full_protocol(protocol)
        cap = float(reservations.get("total_cap_usd"))
        if candidate_total + external_total > cap + 1e-9:
            blockers.append("full-run candidate and external metered reservations exceed the budget cap")
    except (ProtocolValidationError, TypeError, ValueError):
        pass
    return blockers


def _environment_blockers() -> list[str]:
    blockers: list[str] = []
    missing_env = [name for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(name)]
    if missing_env:
        blockers.append(f"missing required provider environment variables: {', '.join(missing_env)}")
    if os.environ.get("ROUTER_BENCHMARK_LLM_CACHE", "").lower() in {"1", "true", "yes", "on"}:
        blockers.append("ROUTER_BENCHMARK_LLM_CACHE must be disabled")
    if os.environ.get("ROUTER_BENCHMARK_TAU2_USE_RESULT_CACHE", "").lower() in {"1", "true", "yes", "on"}:
        blockers.append("ROUTER_BENCHMARK_TAU2_USE_RESULT_CACHE must be disabled")
    package_root = Path(__file__).resolve().parents[1]
    required_paths = [
        package_root / "live" / "tau2env" / "tau2-bench",
        Path.home() / ".local" / "share" / "router_bench_vendor" / "webarena",
    ]
    absent = [str(path) for path in required_paths if not path.exists()]
    if absent:
        blockers.append("missing required local harness paths: " + ", ".join(absent))
    return blockers


def readiness_report(protocol_path: Path, *, check_environment: bool = True) -> dict[str, Any]:
    protocol = load_yaml(protocol_path)
    blockers: list[str] = []
    _check(blockers, "rebuild protocol scope", lambda: validate_rebuild_protocol(protocol, require_approved_budget=False))
    if protocol.get("diagnostic_only") is True:
        blockers.append("diagnostic_only: full run must not be marked diagnostic_only")
    blockers.extend(_budget_blockers(protocol))
    blockers.extend(_execution_identity_blockers(protocol))
    blockers.extend(_reservation_blockers(protocol))
    if check_environment:
        blockers.extend(_environment_blockers())

    candidate_cells = 0
    route_rows = 0
    outcome_rows = 0
    if isinstance(protocol.get("benchmarks"), dict):
        candidate_cells = len(full_run_candidate_cells(protocol))
        route_rows = sum(int(entry.get("total_route_rows", 0)) for entry in protocol["benchmarks"].values())
        outcome_rows = sum(int(entry.get("total_outcome_rows", 0)) for entry in protocol["benchmarks"].values())
    return {
        "status": "ready" if not blockers else "blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_id": protocol.get("protocol_id"),
        "candidate_cells": candidate_cells,
        "route_rows": route_rows,
        "candidate_outcome_rows": outcome_rows,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-environment", action="store_true")
    parser.add_argument("--allow-blockers", action="store_true")
    args = parser.parse_args()
    report = readiness_report(args.protocol, check_environment=not args.skip_environment)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["blockers"] and not args.allow_blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

