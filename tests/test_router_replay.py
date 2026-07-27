from __future__ import annotations

import numpy as np
import pytest

from router_benchmark.interfaces import Candidate, RouteDecision, Router, Task, TaskDomain
from router_benchmark.protocol.router_replay import replay_routes, router_service_cost_usd


class _NoProviderRouter(Router):
    name = "Fixture Router"

    def __init__(self) -> None:
        self.calls = 0

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        assert context == {"replay": True}
        self.calls += 1
        return RouteDecision(
            selected_candidate="cheap-small",
            confidence=0.75,
            fallback_used=False,
            metadata={"router_service_usd": "0.0125", "route_vector": {"kind": "fixture"}},
        )


def _task() -> Task:
    return Task(
        task_id="task-1", benchmark_name="fixture", domain=TaskDomain.QA_REASONING,
        difficulty=0.2, requires_tool_call=False,
        candidates=(Candidate("cheap-small", "cheap", 0, 0, 0),),
    )


def test_router_replay_records_service_cost_without_candidate_execution() -> None:
    router = _NoProviderRouter()
    protocol = {"benchmarks": {"fixture": {"task_ids": ["task-1"], "routing_seed_count": 2}}}
    rows = replay_routes(
        protocol,
        router_configs={"fixture-router": {"router_name": router.name, "package_version": "1.2.3"}},
        routers={"fixture-router": router},
        tasks_by_benchmark={"fixture": [_task()]},
    )

    assert router.calls == 2
    assert [row["router_service_usd"] for row in rows] == ["0.0125", "0.0125"]
    assert all("model_api_cost_usd" not in row for row in rows)
    assert len({row["route_vector_hash"] for row in rows}) == 1


def test_router_service_cost_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        router_service_cost_usd({"router_service_usd": -0.01})
    with pytest.raises(ValueError, match="finite nonnegative"):
        router_service_cost_usd({"router_service_usd": "not-a-cost"})
