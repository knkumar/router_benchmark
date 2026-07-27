"""Paired task and outcome-replicate resampling for canonical analyses."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PairedDraw:
    """One bootstrap draw, reusable across every router comparison."""

    task_replicates: tuple[tuple[str, int], ...]


def paired_task_replicate_draws(
    task_ids: Sequence[str], *, outcome_replicates: int, draws: int, seed: int
) -> list[PairedDraw]:
    """Sample tasks and then one replicate per sampled task with a fixed seed."""
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be nonempty and unique")
    if outcome_replicates < 1 or draws < 1:
        raise ValueError("outcome_replicates and draws must be positive")
    rng = random.Random(seed)
    return [
        PairedDraw(tuple((task_ids[rng.randrange(len(task_ids))], rng.randrange(outcome_replicates)) for _ in task_ids))
        for _ in range(draws)
    ]


def paired_success_difference_interval(
    outcomes: Mapping[tuple[str, str, int], float],
    *,
    first_router: str,
    second_router: str,
    draws: Sequence[PairedDraw],
) -> tuple[float, float, float]:
    """Return point estimate and percentile interval using shared draws.

    ``outcomes`` is keyed by (router_config_id, task_id, outcome_replicate).
    Each draw chooses the same task and replicate for both routers.
    """
    if not draws:
        raise ValueError("draws must be nonempty")
    first_tasks = {task for router, task, _ in outcomes if router == first_router}
    second_tasks = {task for router, task, _ in outcomes if router == second_router}
    if first_tasks != second_tasks or not first_tasks:
        raise ValueError("routers must have identical nonempty task coverage")
    replicate_ids = {replicate for router, _, replicate in outcomes if router == first_router}
    for task in first_tasks:
        for replicate in replicate_ids:
            if (first_router, task, replicate) not in outcomes or (second_router, task, replicate) not in outcomes:
                raise ValueError("routers must have identical task-replicate coverage")

    if any(not 0 <= float(value) <= 1 for value in outcomes.values()):
        raise ValueError("success values must lie in [0, 1]")
    point = sum(float(outcomes[(first_router, task, replicate)]) - float(outcomes[(second_router, task, replicate)]) for task in first_tasks for replicate in replicate_ids) / (len(first_tasks) * len(replicate_ids))
    sampled = []
    for draw in draws:
        values = [
            float(outcomes[(first_router, task, replicate)]) - float(outcomes[(second_router, task, replicate)])
            for task, replicate in draw.task_replicates
        ]
        sampled.append(sum(values) / len(values))
    sampled.sort()
    return point, sampled[int(0.025 * len(sampled))], sampled[min(len(sampled) - 1, int(0.975 * len(sampled)))]


def paired_success_difference_test(
    outcomes: Mapping[tuple[str, str, int], float],
    *,
    first_router: str,
    second_router: str,
    draws: Sequence[PairedDraw],
) -> tuple[float, float, float, float]:
    """Task-clustered paired inference: point estimate, 95% percentile interval,
    and a two-sided cluster-bootstrap p-value for H0: mean difference = 0.

    The interval and the p-value are read off the *same* bootstrap distribution
    (drawn over tasks, one replicate per sampled task), so a 95% interval that
    excludes zero always corresponds to p < 0.05 and vice versa. This is the
    unit-consistent replacement for a row-level McNemar test, which would treat
    the routing trials and outcome replicates nested inside a task as
    independent observations and understate uncertainty.
    """
    point, low, high = paired_success_difference_interval(
        outcomes, first_router=first_router, second_router=second_router, draws=draws
    )
    first_tasks = {task for router, task, _ in outcomes if router == first_router}
    replicate_ids = {replicate for router, _, replicate in outcomes if router == first_router}
    sampled = sorted(
        sum(
            float(outcomes[(first_router, task, replicate)]) - float(outcomes[(second_router, task, replicate)])
            for task, replicate in draw.task_replicates
        )
        / len(draw.task_replicates)
        for draw in draws
    )
    total = len(sampled)
    at_or_below = sum(1 for value in sampled if value <= 0.0)
    at_or_above = sum(1 for value in sampled if value >= 0.0)
    # Add-one smoothing keeps the p-value strictly positive and bounded away
    # from an impossible exact zero at finite draw counts.
    left = (at_or_below + 1) / (total + 1)
    right = (at_or_above + 1) / (total + 1)
    p_value = min(1.0, 2.0 * min(left, right))
    return point, low, high, p_value
