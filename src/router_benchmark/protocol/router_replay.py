"""Replay frozen router decisions without executing candidate models.

Candidate generation is intentionally outside this module.  A replay records
the route selected for each frozen task and any metered router-service spend in
``router_service_usd``.  That field is never folded into the candidate
``model_api_cost_usd`` reported by the canonical outcome matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from typing import Any, Mapping, Sequence

import numpy as np

from router_benchmark.interfaces import Router, Task


def _seed_from(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8")) & 0xFFFFFFFF


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def router_service_cost_usd(metadata: Mapping[str, Any]) -> float:
    """Read separately-metered router cost from an adapter decision.

    Adapters that route locally simply omit the value and record zero.  An
    adapter that calls a metered routing service must return its measured cost
    under ``router_service_usd``; candidate calls are not accepted here.
    """
    value = metadata.get("router_service_usd", 0.0)
    try:
        cost = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("router_service_usd must be a finite nonnegative number") from exc
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("router_service_usd must be a finite nonnegative number")
    return cost


def replay_routes(
    protocol: Mapping[str, Any],
    *,
    router_configs: Mapping[str, Mapping[str, Any]],
    routers: Mapping[str, Router],
    tasks_by_benchmark: Mapping[str, Sequence[Task]],
) -> list[dict[str, str]]:
    """Return canonical route rows for frozen tasks and routing seeds.

    This function invokes only ``Router.route``.  It never calls benchmark
    scoring or candidate-generation clients, so callers can execute candidate
    generation once and replay routes independently.  Tests should use fake
    routers, while live adapters may place a measured service charge in route
    decision metadata.
    """
    if set(router_configs) != set(routers):
        raise ValueError("router_configs and routers must have identical IDs")

    rows: list[dict[str, str]] = []
    for benchmark_id, benchmark_spec in protocol["benchmarks"].items():
        tasks = {str(task.task_id): task for task in tasks_by_benchmark[benchmark_id]}
        expected_ids = [str(task_id) for task_id in benchmark_spec["task_ids"]]
        if set(tasks) != set(expected_ids):
            raise ValueError(f"frozen task IDs do not match protocol for {benchmark_id}")
        for router_config_id, config in router_configs.items():
            router = routers[router_config_id]
            if config.get("router_name") != router.name:
                raise ValueError(f"router config name mismatch for {router_config_id}")
            for task_id in expected_ids:
                task = tasks[task_id]
                for routing_seed in range(benchmark_spec["routing_seed_count"]):
                    rng = np.random.default_rng(_seed_from(benchmark_id, task_id, router_config_id, str(routing_seed)))
                    started = time.monotonic()
                    decision = router.route(task, context={"replay": True}, rng=rng)
                    latency_ms = (time.monotonic() - started) * 1000.0
                    if decision.selected_candidate not in {candidate.name for candidate in task.candidates}:
                        raise ValueError(f"router selected undeclared candidate for {benchmark_id}/{task_id}")
                    metadata = decision.metadata
                    service_cost = router_service_cost_usd(metadata)
                    route_vector = {
                        "selected_candidate": decision.selected_candidate,
                        "fallback_used": decision.fallback_used,
                        "metadata": metadata.get("route_vector", metadata),
                    }
                    rows.append({
                        "router_config_id": router_config_id,
                        "benchmark_id": benchmark_id,
                        "task_id": task_id,
                        "routing_seed": str(routing_seed),
                        "selected_candidate": decision.selected_candidate,
                        "confidence": str(decision.confidence),
                        "fallback_path": str(metadata.get("fallback_path", "declared" if decision.fallback_used else "none")),
                        "decision_latency_ms": str(latency_ms),
                        "router_service_usd": str(service_cost),
                        "package_version": str(config.get("package_version", "unknown")),
                        "configuration_digest": _digest(config),
                        "route_vector_hash": _digest(route_vector),
                    })
    return rows
