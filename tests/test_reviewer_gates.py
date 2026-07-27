from __future__ import annotations

import csv
import json
from pathlib import Path

from router_benchmark.analysis.reviewer_gates import run_reviewer_gates
from test_outcome_matrix import _build_bundle


def test_reviewer_gates_write_baseline_bfcl_and_ablation_artifacts(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    output = tmp_path / "analysis"

    run_reviewer_gates(bundle, protocol, output)

    candidate_rows = list(csv.DictReader((output / "candidate_tier_summary.csv").open(encoding="utf-8")))
    baseline_rows = list(csv.DictReader((output / "baseline_summary.csv").open(encoding="utf-8")))
    bfcl_rows = list(csv.DictReader((output / "bfcl_route_equivalence.csv").open(encoding="utf-8")))
    reconciliation = json.loads((output / "baseline_reconciliation.json").read_text(encoding="utf-8"))
    ablation = json.loads((output / "ablation_registry.json").read_text(encoding="utf-8"))

    assert len(candidate_rows) == 12
    assert {row["baseline_id"] for row in baseline_rows} == {
        "Always-Cheapest Baseline (live)",
        "Always-Mid Baseline (live)",
        "Always-Strongest Baseline (live)",
    }
    assert reconciliation["status"] == "passed"
    assert all(row["equivalent_outcome"] == "true" for row in bfcl_rows)
    assert ablation["status"] == "deferred"

