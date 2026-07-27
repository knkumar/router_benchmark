from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from router_benchmark.interfaces import Candidate, RouteDecision, Task, TaskDomain
from router_benchmark.protocol.dry_run_execution import run_dry_candidate_stage
from test_dry_run_preflight import _dry_from, _protocol


class FixtureAdapter:
    def __init__(self, benchmark_id: str, task_ids: list[str]) -> None:
        self.name = benchmark_id
        self.tasks = [self._task(benchmark_id, task_id) for task_id in reversed(task_ids)]
        self.scored: list[tuple[str, str, dict]] = []

    @staticmethod
    def _task(benchmark_id: str, task_id: str) -> Task:
        candidates = tuple(
            Candidate(candidate_id, candidate_id, 0.0, 0.0, 0.0)
            for candidate_id in ("cheap-small", "mid-general", "strong-frontier")
        )
        return Task(task_id, benchmark_id, TaskDomain.QA_REASONING, 0.0, False, candidates)

    def generate_tasks(self, rng) -> list[Task]:
        return self.tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        self.scored.append((task.task_id, decision.selected_candidate, decision.metadata))
        assert decision.metadata["candidate_execution"] == "forced_frozen_candidate"
        assert decision.metadata["router_replay"] is False
        return {"success": True, "cost_usd": 0.1, "latency_ms": 2.0}


def _adapters(dry: dict) -> dict[str, FixtureAdapter]:
    return {
        benchmark_id: FixtureAdapter(benchmark_id, entry["task_ids"])
        for benchmark_id, entry in dry["benchmarks"].items()
    }


def test_dry_execution_selects_frozen_ids_and_forces_every_candidate_without_router_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    adapters = _adapters(dry)
    import router_benchmark.protocol.router_replay as router_replay

    monkeypatch.setattr(router_replay, "replay_routes", lambda *args, **kwargs: pytest.fail("router replay called"))
    rows = run_dry_candidate_stage(
        dry, frozen, stage_dir=tmp_path / "stage", adapters=adapters, estimate_cost_usd=lambda _: 0.1
    )

    assert len(rows) == 12
    for benchmark_id, adapter in adapters.items():
        expected_task = dry["benchmarks"][benchmark_id]["task_ids"][0]
        assert [task_id for task_id, _, _ in adapter.scored] == [expected_task] * 3
        assert [candidate_id for _, candidate_id, _ in adapter.scored] == ["cheap-small", "mid-general", "strong-frontier"]


def test_invalid_preflight_does_not_touch_adapter_mapping(tmp_path: Path) -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    dry["dry_run_budget_cap_usd"] = 10.01
    adapters = _adapters(_dry_from(frozen))
    with pytest.raises(ValueError, match="budget_cap"):
        run_dry_candidate_stage(
            dry, frozen, stage_dir=tmp_path / "stage", adapters=adapters, estimate_cost_usd=lambda _: 0.1
        )
    assert all(adapter.scored == [] for adapter in adapters.values())
