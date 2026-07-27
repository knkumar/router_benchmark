from __future__ import annotations

import pytest

from router_benchmark.interfaces import Router, Task, TaskDomain
from router_benchmark.protocol.production_adapters import _EstimatedMeteredRouter, build_dry_run_adapters


class _Adapter:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs


def _dry_protocol() -> dict:
    return {
        "benchmarks": {
            "RouterBench (live)": {"task_ids": ["routerbench-0000"]},
            "BFCL v4 (live)": {"task_ids": ["bfcl-0000-simple_java_51"]},
            "tau2-bench (live)": {"task_ids": ["tau2-0"]},
            "WebArena (live)": {"task_ids": ["webarena-102"]},
        },
        "dry_run_cost_reservations": {
            "request_limits": {
                "RouterBench (live)": {"max_provider_calls_per_candidate_cell": 0, "max_output_tokens_per_call": 0, "max_steps": 0},
                "BFCL v4 (live)": {"max_provider_calls_per_candidate_cell": 1, "max_output_tokens_per_call": 256, "max_steps": 1},
                "tau2-bench (live)": {"max_provider_calls_per_candidate_cell": 60, "max_output_tokens_per_call": 1024, "max_steps": 30},
                "WebArena (live)": {"max_provider_calls_per_candidate_cell": 30, "max_output_tokens_per_call": 384, "max_steps": 30},
            },
        },
    }


def test_factory_uses_allowlisted_adapters_and_historical_sample_sizes(monkeypatch) -> None:
    import router_benchmark.live.live_benchmarks as live_benchmarks
    import router_benchmark.live.tau2_live as tau2_live
    import router_benchmark.live.webarena_live as webarena_live

    monkeypatch.setattr(live_benchmarks, "RouterBenchLive", lambda **kwargs: _Adapter("RouterBench (live)", **kwargs))
    monkeypatch.setattr(live_benchmarks, "BFCLLive", lambda **kwargs: _Adapter("BFCL v4 (live)", **kwargs))
    monkeypatch.setattr(tau2_live, "Tau2BenchLive", lambda **kwargs: _Adapter("tau2-bench (live)", **kwargs))
    monkeypatch.setattr(webarena_live, "WebArenaLive", lambda **kwargs: _Adapter("WebArena (live)", **kwargs))

    adapters = build_dry_run_adapters(_dry_protocol())

    assert adapters["RouterBench (live)"].kwargs == {"n_tasks": 60, "task_ids": ["routerbench-0000"]}
    assert adapters["BFCL v4 (live)"].kwargs == {"n_tasks": 30, "task_ids": ["bfcl-0000-simple_java_51"]}
    assert adapters["tau2-bench (live)"].kwargs == {
        "task_ids": ["tau2-0"],
        "max_steps": 30,
        "max_output_tokens": 1024,
        "require_cost_ledger": True,
    }
    assert adapters["WebArena (live)"].kwargs == {
        "task_ids": ["webarena-102"],
        "max_steps": 30,
        "max_output_tokens": 384,
        "require_trace_cost": True,
    }


class _FailingRouter(Router):
    name = "Failing Router"

    def route(self, task, context, rng):
        raise RuntimeError("service unavailable")


def test_metered_router_records_fallback_row_on_route_error() -> None:
    wrapped = _EstimatedMeteredRouter(
        _FailingRouter(),
        router_service_usd=0.05,
        metering_basis="fixture",
    )
    task = Task("task-1", "fixture", TaskDomain.QA_REASONING, 0.0, False, ())

    decision = wrapped.route(task, context={}, rng=None)

    assert decision.selected_candidate == "mid-general"
    assert decision.confidence == 0.0
    assert decision.fallback_used is True
    assert decision.metadata["router_service_usd"] == pytest.approx(0.05)
    assert decision.metadata["fallback_path"] == "router_service_error"
