"""Evaluation harness: routers x benchmarks -> results.

This is deliverable (1): "write an interface which takes in a list of
routers and a list of benchmarks, then evaluates each router against each
benchmark."

Usage:
    from router_benchmark.harness import EvaluationHarness
    from router_benchmark.routers import build_all_routers
    from router_benchmark.benchmarks import build_all_benchmarks

    harness = EvaluationHarness(seed=1234, n_trials=3)
    results_df = harness.evaluate(build_all_routers(), build_all_benchmarks())

`results_df` is a tidy pandas DataFrame with one row per
(router, benchmark, task, trial) -- see interfaces.TaskResult for columns.
Feed it to metrics.compute_all_metrics() for the comparison table, and to
plots.py to render figures.
"""

from __future__ import annotations

import time
import zlib
from typing import Callable

import numpy as np
import pandas as pd

from router_benchmark.interfaces import Benchmark, Router, TaskResult


def _seed_from(*parts: str) -> int:
    """Deterministic 32-bit seed derived from arbitrary string parts, so the
    same (router, benchmark, trial) always reproduces identical results."""
    key = "|".join(parts).encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


class EvaluationHarness:
    """Evaluates every router against every benchmark.

    n_trials > 1 repeats each (router, benchmark) pass with an independently
    seeded RNG so metrics.py can measure route stability (does the router
    make the same decision on the same task across repeated trials?) and
    give variance-aware estimates of every other metric.
    """

    def __init__(self, seed: int = 1234, n_trials: int = 3):
        self.seed = seed
        self.n_trials = n_trials

    def evaluate(
        self,
        routers: list[Router],
        benchmarks: list[Benchmark],
        on_row_complete: Callable[[TaskResult], None] | None = None,
        completed_keys: set[tuple[str, str, str, str]] | None = None,
    ) -> pd.DataFrame:
        """`on_row_complete`, if given, is called with each TaskResult the
        instant it's computed -- before the whole run finishes. This lets a
        caller persist results incrementally (see run_common.py) so a
        mid-run crash doesn't lose every row computed so far, only the
        in-memory `rows` list this method still builds and returns.

        `completed_keys`, if given, is a set of already-finished
        (router_name, benchmark_name, task_id, trial) tuples (all strings) to
        skip -- used only to resume an interrupted run. The caller owns this
        set; the harness never reads any file or output directory itself, so
        it stays a pure, reusable component. Skipped tasks are absent from the
        returned DataFrame, so a resuming caller should read the full result
        set from its own persisted store, not this return value alone."""
        completed_keys = completed_keys or set()
        rows: list[TaskResult] = []

        for benchmark in benchmarks:
            task_gen_rng = np.random.default_rng(_seed_from("tasks", str(self.seed), benchmark.name))
            tasks = benchmark.generate_tasks(task_gen_rng)

            for router in routers:
                for trial in range(self.n_trials):
                    trial_seed = _seed_from(router.name, benchmark.name, str(trial), str(self.seed))
                    rng = np.random.default_rng(trial_seed)

                    for task in tasks:
                        if (router.name, benchmark.name, str(task.task_id), str(trial)) in completed_keys:
                            continue
                        _route_start = time.monotonic()
                        decision = router.route(task, context={}, rng=rng)
                        router_decision_latency_ms = (time.monotonic() - _route_start) * 1000.0
                        # Mutating the metadata dict (not reassigning the
                        # frozen dataclass field) so benchmark.score() can
                        # attach full trace context (router/trial) to any
                        # real API calls it makes for reproducibility.
                        decision.metadata["router_name"] = router.name
                        decision.metadata["trial"] = trial
                        outcome = benchmark.score(task, decision, rng)

                        result = TaskResult(
                            router_name=router.name,
                            benchmark_name=benchmark.name,
                            task_id=task.task_id,
                            domain=task.domain,
                            difficulty=task.difficulty,
                            trial=trial,
                            selected_candidate=decision.selected_candidate,
                            confidence=decision.confidence,
                            fallback_used=decision.fallback_used,
                            success=outcome["success"],
                            tool_call_required=task.requires_tool_call,
                            tool_call_correct=outcome["tool_call_correct"],
                            cost_usd=outcome["cost_usd"],
                            latency_ms=outcome["latency_ms"],
                            router_decision_latency_ms=router_decision_latency_ms,
                        )
                        rows.append(result)
                        if on_row_complete is not None:
                            on_row_complete(result)

        if not rows:
            # Every task was skipped via completed_keys (a fully-resumed run).
            # Return an empty frame with the correct columns rather than
            # crashing on a missing "domain" column downstream.
            return pd.DataFrame(columns=list(TaskResult.__dataclass_fields__))
        df = pd.DataFrame([r.__dict__ for r in rows])
        df["domain"] = df["domain"].apply(lambda d: d.value if hasattr(d, "value") else d)
        return df
