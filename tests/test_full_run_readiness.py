from __future__ import annotations

from pathlib import Path

from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.preflight_full_run import (
    candidate_reservations_from_full_protocol,
    external_metered_reservation_from_full_protocol,
)
from router_benchmark.scripts.audit_full_run_readiness import readiness_report
from test_full_run_preflight import _full_protocol


def test_current_full_protocol_readiness_is_ready_without_environment_checks() -> None:
    report = readiness_report(Path("protocol/paper1_rebuild.yaml"), check_environment=False)

    assert report["status"] == "ready"
    assert report["candidate_cells"] == 2610
    assert report["route_rows"] == 2320
    assert report["blockers"] == []


def test_approved_fixture_readiness_is_ready_without_environment_checks(tmp_path: Path) -> None:
    protocol = _full_protocol()
    path = tmp_path / "protocol.yaml"
    import yaml

    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

    report = readiness_report(path, check_environment=False)

    assert report["status"] == "ready"
    assert report["blockers"] == []
    assert report["candidate_cells"] == 24


def test_current_full_protocol_readiness_does_not_mutate_protocol() -> None:
    before = Path("protocol/paper1_rebuild.yaml").read_text(encoding="utf-8")
    readiness_report(Path("protocol/paper1_rebuild.yaml"), check_environment=False)
    after = Path("protocol/paper1_rebuild.yaml").read_text(encoding="utf-8")
    assert after == before
    assert load_yaml(Path("protocol/paper1_rebuild.yaml"))["execution_budget"]["approval_status"] == "approved"


def test_current_full_protocol_reservations_match_planned_cost_envelope() -> None:
    protocol = load_yaml(Path("protocol/paper1_rebuild.yaml"))
    reservations = candidate_reservations_from_full_protocol(protocol)

    assert len(reservations) == 2610
    assert round(sum(reservations.values()), 2) == 1884.30
    assert external_metered_reservation_from_full_protocol(protocol) == 65.20
    assert protocol["full_run_cost_reservations"]["total_cap_usd"] == 2500

