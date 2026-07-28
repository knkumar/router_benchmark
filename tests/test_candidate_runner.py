from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from router_benchmark.protocol.candidate_runner import CandidateBudgetExceeded, CandidateStageRunner


def _protocol() -> dict:
    return {
        "candidates": ["cheap-small", "mid-general", "strong-frontier"],
        "pricing": {"as_of": "fixture-price"},
        "model_snapshots": {candidate: "fixture-model" for candidate in ("cheap-small", "mid-general", "strong-frontier")},
        "grader_versions": {"fixture-benchmark": "fixture-grader"},
        "benchmarks": {
            "fixture-benchmark": {
                "task_ids": ["task-1"],
                "outcome_replicates_per_task_candidate": 1,
            }
        },
    }


def _response(cell) -> dict:
    return {
        "success": cell.candidate_id != "cheap-small",
        "provider_generation_usd": 1.0,
        "fallback_generation_usd": 0.0,
        "generation_latency_ms": 4.0,
        "raw_trace": {"candidate": cell.candidate_id, "response": "fixture"},
    }


def test_staged_candidate_execution_writes_canonical_rows_and_trace_lineage(tmp_path: Path) -> None:
    runner = CandidateStageRunner(tmp_path / "stage", _protocol(), budget_cap_usd=10.0)
    rows = runner.run(_response, lambda _: 1.0)

    assert len(rows) == 3
    with (tmp_path / "stage" / "candidate_outcomes.csv").open(encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    traces = [json.loads(line) for line in (tmp_path / "stage" / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["raw_trace_digest"] for row in persisted} == {trace["digest"] for trace in traces}
    assert all(row["cache_flag"] == "false" for row in persisted)
    assert all(row["model_api_cost_usd"] == "1.0" for row in persisted)


def test_resume_skips_persisted_candidate_cells(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    first = CandidateStageRunner(stage, _protocol(), budget_cap_usd=10.0)
    calls: list[str] = []

    def interrupted(cell):
        calls.append(cell.candidate_id)
        if len(calls) == 2:
            raise RuntimeError("fixture interruption")
        return _response(cell)

    with pytest.raises(RuntimeError, match="interruption"):
        first.run(interrupted, lambda _: 1.0)
    resumed_calls: list[str] = []
    CandidateStageRunner(stage, _protocol(), budget_cap_usd=10.0).run(
        lambda cell: (resumed_calls.append(cell.candidate_id), _response(cell))[1], lambda _: 1.0, resume=True
    )
    assert calls == ["cheap-small", "mid-general"]
    assert resumed_calls == ["mid-general", "strong-frontier"]
    with (stage / "candidate_outcomes.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_runner_enforces_ten_dollar_cap_before_next_call(tmp_path: Path) -> None:
    runner = CandidateStageRunner(tmp_path / "stage", _protocol(), budget_cap_usd=10.0)
    calls: list[str] = []
    with pytest.raises(CandidateBudgetExceeded, match=r"\$10.00 cap"):
        runner.run(lambda cell: (calls.append(cell.candidate_id), _response(cell))[1], lambda _: 11.0)
    assert calls == []


def test_runner_rejects_reservations_that_do_not_cover_exact_cells(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        CandidateStageRunner(
            tmp_path / "stage",
            _protocol(),
            budget_cap_usd=10.0,
            reservation_by_cell={"fixture-benchmark|task-1|cheap-small|0": 1.0},
        )


def test_runner_reserves_pending_cells_and_external_spend_before_call(tmp_path: Path) -> None:
    reservations = {
        f"fixture-benchmark|task-1|{candidate}|0": 3.1
        for candidate in ("cheap-small", "mid-general", "strong-frontier")
    }
    with pytest.raises(CandidateBudgetExceeded, match="reserved"):
        CandidateStageRunner(
            tmp_path / "stage",
            _protocol(),
            budget_cap_usd=10.0,
            reservation_by_cell=reservations,
            external_reserved_usd=1.0,
        )


def test_runner_rejects_nonfinite_estimate_before_execution(tmp_path: Path) -> None:
    runner = CandidateStageRunner(tmp_path / "stage", _protocol(), budget_cap_usd=10.0)
    with pytest.raises(ValueError, match="finite nonnegative"):
        runner.run(_response, lambda _: float("nan"))


def test_resume_rejects_nonfinite_persisted_cost(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    runner = CandidateStageRunner(stage, _protocol(), budget_cap_usd=10.0)
    runner.run(_response, lambda _: 1.0)
    header = (
        "benchmark_id,task_id,candidate_id,outcome_replicate,execution_seed,grader_version,"
        "raw_trace_digest,success,provider_generation_usd,fallback_generation_usd,model_api_cost_usd,"
        "generation_latency_ms,failure_status,cache_flag,model_version,pricing_snapshot\n"
    )
    row = "fixture-benchmark,task-1,cheap-small,0,0,fixture,trace,true,nan,0,nan,1,none,false,fixture,fixture\n"
    (stage / "candidate_outcomes.csv").write_text(header + row, encoding="utf-8")
    (stage / "traces.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        runner.run(_response, lambda _: 1.0, resume=True)


def test_resume_rejects_stage_manifest_mismatch(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    CandidateStageRunner(stage, _protocol(), budget_cap_usd=10.0).run(_response, lambda _: 1.0)
    changed = _protocol()
    changed["pricing"]["as_of"] = "changed-price"
    with pytest.raises(ValueError, match="manifest"):
        CandidateStageRunner(stage, changed, budget_cap_usd=10.0).run(_response, lambda _: 1.0, resume=True)


def test_stage_rejects_locked_bundle_directory(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text("{}", encoding="utf-8")
    runner = CandidateStageRunner(stage, _protocol(), budget_cap_usd=10.0)
    with pytest.raises(ValueError, match="locked canonical bundle"):
        runner.run(_response, lambda _: 1.0, resume=True)


def test_stage_rejects_dirty_directory_without_resume(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "leftover.txt").write_text("old run", encoding="utf-8")
    runner = CandidateStageRunner(stage, _protocol(), budget_cap_usd=10.0)
    with pytest.raises(ValueError, match="empty unless --resume"):
        runner.run(_response, lambda _: 1.0)


def test_runner_persists_external_metered_spend_ledger(tmp_path: Path) -> None:
    protocol = _protocol()
    reservations = {
        f"fixture-benchmark|task-1|{candidate}|0": 0.1
        for candidate in protocol["candidates"]
    }

    def response_with_external_spend(cell):
        response = _response(cell)
        response["provider_generation_usd"] = 0.1
        response["external_metered_usd"] = 0.2
        return response

    stage = tmp_path / "stage"
    CandidateStageRunner(
        stage,
        protocol,
        budget_cap_usd=1.0,
        reservation_by_cell=reservations,
        external_reserved_usd=0.6,
    ).run(response_with_external_spend, lambda _: 0.1)

    records = [json.loads(line) for line in (stage / "external_metered_spend.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert sum(record["external_metered_usd"] for record in records) == pytest.approx(0.6)
    CandidateStageRunner(
        stage,
        protocol,
        budget_cap_usd=1.0,
        reservation_by_cell=reservations,
        external_reserved_usd=0.6,
    ).run(response_with_external_spend, lambda _: 0.1, resume=True)


def test_runner_rejects_external_spend_over_reserved_budget(tmp_path: Path) -> None:
    protocol = _protocol()
    reservations = {
        f"fixture-benchmark|task-1|{candidate}|0": 0.1
        for candidate in protocol["candidates"]
    }

    def response_with_external_overage(cell):
        response = _response(cell)
        response["provider_generation_usd"] = 0.1
        response["external_metered_usd"] = 0.7
        return response

    with pytest.raises(CandidateBudgetExceeded, match="external metered spend"):
        CandidateStageRunner(
            tmp_path / "stage",
            protocol,
            budget_cap_usd=1.0,
            reservation_by_cell=reservations,
            external_reserved_usd=0.6,
        ).run(response_with_external_overage, lambda _: 0.1)


def test_resume_rejects_external_ledger_without_completed_row(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    reservations = {
        f"fixture-benchmark|task-1|{candidate}|0": 1.0
        for candidate in _protocol()["candidates"]
    }
    runner = CandidateStageRunner(
        stage,
        _protocol(),
        budget_cap_usd=10.0,
        reservation_by_cell=reservations,
        external_reserved_usd=1.0,
    )

    def expensive_external(cell):
        response = _response(cell)
        response["external_metered_usd"] = 2.0
        return response

    with pytest.raises(CandidateBudgetExceeded, match="external metered"):
        runner.run(expensive_external, lambda _: 1.0)
    with pytest.raises(ValueError, match="without a completed candidate row"):
        runner.run(_response, lambda _: 1.0, resume=True)
