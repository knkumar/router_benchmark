"""Frozen task selection remains local, exact, and provider-free."""

from __future__ import annotations

import pandas as pd
import pytest

from router_benchmark.live.live_benchmarks import BFCLLive, RouterBenchLive, _ROUTERBENCH_TIER_MODEL
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.webarena_live import WebArenaLive


def test_routerbench_uses_requested_frozen_ids_in_requested_order(monkeypatch: pytest.MonkeyPatch) -> None:
    models = list(_ROUTERBENCH_TIER_MODEL.values())
    rows = []
    for index in range(3):
        row = {"prompt": f"prompt {index}", "eval_name": "fixture"}
        row.update({model: 1.0 for model in models})
        row.update({f"{model}|total_cost": 0.01 for model in models})
        rows.append(row)
    adapter = RouterBenchLive(task_ids=["routerbench-0002", "routerbench-0000"])
    monkeypatch.setattr(adapter, "_load", lambda: pd.DataFrame(rows))

    tasks = adapter.generate_tasks(rng=None)

    assert [task.task_id for task in tasks] == ["routerbench-0002", "routerbench-0000"]
    assert [task.metadata["prompt"] for task in tasks] == ["prompt 2", "prompt 0"]


def test_bfcl_uses_frozen_id_and_rejects_unavailable_id(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        {"question": {"id": "simple_python_7", "function": [{"name": "f", "parameters": {"required": []}}], "question": [[{"content": "first"}]]}, "ground_truth": [{}]},
        {"question": {"id": "simple_java_51", "function": [{"name": "g", "parameters": {"required": []}}], "question": [[{"content": "second"}]]}, "ground_truth": [{}]},
    ]
    adapter = BFCLLive(n_tasks=2, task_ids=["bfcl-0000-simple_java_51"])
    monkeypatch.setattr(adapter, "_load_raw", lambda: items)

    assert [task.task_id for task in adapter.generate_tasks(rng=None)] == ["bfcl-0000-simple_java_51"]

    invalid = BFCLLive(n_tasks=2, task_ids=["bfcl-0000-not-present"])
    monkeypatch.setattr(invalid, "_load_raw", lambda: items)
    with pytest.raises(ValueError, match="unavailable"):
        invalid.generate_tasks(rng=None)


def test_tau2_uses_exact_frozen_ids_without_running_harness() -> None:
    adapter = Tau2BenchLive.__new__(Tau2BenchLive)
    adapter.n_tasks = 1
    adapter.domain = "retail"
    adapter._frozen_task_ids = ("tau2-2", "tau2-0")
    adapter._all_tasks = [_tau2_task(0), _tau2_task(2)]

    tasks = adapter.generate_tasks(rng=None)

    assert [task.task_id for task in tasks] == ["tau2-2", "tau2-0"]


def test_webarena_uses_exact_frozen_ids_and_rejects_duplicates() -> None:
    adapter = WebArenaLive.__new__(WebArenaLive)
    adapter.n_tasks = 1
    adapter._frozen_task_ids = ("webarena-12", "webarena-3")
    adapter._pool = [_webarena_task(3), _webarena_task(12)]

    tasks = adapter.generate_tasks(rng=None)

    assert [task.task_id for task in tasks] == ["webarena-12", "webarena-3"]
    with pytest.raises(ValueError, match="duplicates"):
        WebArenaLive(n_tasks=1, task_ids=["webarena-3", "webarena-3"])


def _tau2_task(task_id: int) -> dict:
    return {
        "id": task_id,
        "user_scenario": {"instructions": {"reason_for_call": "fixture"}},
        "evaluation_criteria": {"actions": []},
    }


def _webarena_task(task_id: int) -> dict:
    return {"task_id": task_id, "sites": ["gitlab"], "intent": "fixture", "eval": {"eval_types": []}}
