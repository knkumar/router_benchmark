from __future__ import annotations

from pathlib import Path


ROOT = Path.cwd()


def test_makefile_exposes_protocol_execution_entrypoints() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "validate-canonical:",
        "dry-run-preflight:",
        "dry-run-candidates:",
        "dry-run-routes:",
        "dry-run-bundle:",
        "full-run-preflight:",
        "full-run-readiness:",
        "full-run-approval-packet:",
        "full-run-status:",
        "full-run-spend-summary:",
        "full-run-candidates:",
        "full-run-routes:",
        "full-run-bundle:",
        "rebuild-analysis:",
        "reviewer-gates:",
    ):
        assert target in makefile


def test_packaged_analysis_uses_canonical_bundle_inputs() -> None:
    active = (ROOT / "src/router_benchmark/analysis/paired_tests.py").read_text(encoding="utf-8")
    assert "output/results.csv" not in active
    assert "validate_bundle" in active
