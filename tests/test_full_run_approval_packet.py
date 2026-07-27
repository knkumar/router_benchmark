from __future__ import annotations

import json
from pathlib import Path

from router_benchmark.scripts.generate_full_run_approval_packet import build_packet


def test_full_run_approval_packet_summarizes_cap_and_does_not_approve_execution(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps({
            "status": "blocked",
            "blockers": ["execution_budget.approval_status: must be approved before full execution"],
        }),
        encoding="utf-8",
    )

    packet = build_packet(Path("protocol/paper1_rebuild.yaml"), readiness)

    assert "This packet is a review artifact only" in packet
    assert "Current approval status: `approved`" in packet
    assert "Hard cap: $2,500.00" in packet
    assert "Candidate model API reserve: $1,884.30 across 2610 candidate cells" in packet
    assert "Router-service reserve: $40.60 across 2320 route rows" in packet
    assert "Noncandidate external metered reserve: $24.60" in packet
    assert "approval_status: approved" in packet
    assert "PENDING" not in packet


def test_full_run_approval_packet_does_not_mutate_protocol(tmp_path: Path) -> None:
    protocol = Path("protocol/paper1_rebuild.yaml")
    before = protocol.read_text(encoding="utf-8")
    build_packet(protocol)
    after = protocol.read_text(encoding="utf-8")
    assert after == before

