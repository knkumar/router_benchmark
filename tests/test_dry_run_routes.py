from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from router_benchmark.interfaces import RouteDecision, Router, Task
from router_benchmark.protocol.dry_run_routes import run_dry_route_stage
from test_dry_run_execution import _adapters
from test_dry_run_preflight import _dry_from, _protocol


class _MeteredRouter(Router):
    def __init__(self, name: str, cost: float = 0.01) -> None:
        self.name = name
        self.cost = cost
        self.calls = 0

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        self.calls += 1
        return RouteDecision(
            selected_candidate="cheap-small",
            confidence=0.5,
            fallback_used=False,
            metadata={"router_service_usd": self.cost, "route_vector": {"task": task.task_id}},
        )


def _write_completed_candidate_stage(stage: Path, dry: dict) -> None:
    stage.mkdir()
    (stage / "stage_manifest.json").write_text("{}\n", encoding="utf-8")
    (stage / "traces.jsonl").write_text("{}\n", encoding="utf-8")
    rows = []
    for benchmark, entry in dry["benchmarks"].items():
        for task_id in entry["task_ids"]:
            for candidate in dry["candidates"]:
                rows.append(
                    {
                        "benchmark_id": benchmark,
                        "task_id": task_id,
                        "candidate_id": candidate,
                        "outcome_replicate": "0",
                        "execution_seed": "0",
                        "grader_version": "fixture",
                        "raw_trace_digest": "fixture",
                        "success": "true",
                        "provider_generation_usd": "0.01",
                        "fallback_generation_usd": "0.0",
                        "model_api_cost_usd": "0.01",
                        "generation_latency_ms": "1.0",
                        "failure_status": "none",
                        "cache_flag": "false",
                        "model_version": "fixture",
                        "pricing_snapshot": "fixture",
                    }
                )
    with (stage / "candidate_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _routers(dry: dict, cost: float = 0.01):
    routers = {name.lower().replace(" ", "-"): _MeteredRouter(name, cost) for name in dry["routers"]}
    configs = {
        router_id: {"router_name": router.name, "package_version": "fixture"}
        for router_id, router in routers.items()
    }
    return configs, routers


def test_dry_route_stage_writes_routes_and_enforces_no_rerun(tmp_path: Path) -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    stage = tmp_path / "stage"
    _write_completed_candidate_stage(stage, dry)
    configs, routers = _routers(dry)

    rows = run_dry_route_stage(
        dry,
        frozen,
        stage_dir=stage,
        adapters=_adapters(dry),
        router_configs=configs,
        routers=routers,
    )

    assert len(rows) == 16
    assert sum(float(row["router_service_usd"]) for row in rows) == pytest.approx(0.16)
    assert (stage / "routes.csv").exists()
    assert (stage / "router_configs.json").exists()
    assert all(router.calls == 4 for router in routers.values())
    with pytest.raises(ValueError, match="already exist"):
        run_dry_route_stage(
            dry,
            frozen,
            stage_dir=stage,
            adapters=_adapters(dry),
            router_configs=configs,
            routers=routers,
        )


def test_dry_route_stage_rejects_router_service_over_reservation(tmp_path: Path) -> None:
    frozen = _protocol()
    dry = _dry_from(frozen)
    stage = tmp_path / "stage"
    _write_completed_candidate_stage(stage, dry)
    configs, routers = _routers(dry, cost=0.5)

    with pytest.raises(RuntimeError, match="router-service spend"):
        run_dry_route_stage(
            dry,
            frozen,
            stage_dir=stage,
            adapters=_adapters(dry),
            router_configs=configs,
            routers=routers,
        )
