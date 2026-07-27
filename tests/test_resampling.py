from __future__ import annotations

from router_benchmark.analysis.resampling import paired_success_difference_interval, paired_task_replicate_draws


def test_paired_draws_are_byte_stable_for_fixed_seed() -> None:
    first = paired_task_replicate_draws(["t1", "t2", "t3"], outcome_replicates=2, draws=20, seed=9)
    second = paired_task_replicate_draws(["t1", "t2", "t3"], outcome_replicates=2, draws=20, seed=9)
    assert first == second


def test_replicate_disagreement_changes_paired_interval() -> None:
    draws = paired_task_replicate_draws(["t1", "t2"], outcome_replicates=2, draws=1000, seed=11)
    outcomes = {
        ("a", "t1", 0): True, ("a", "t1", 1): False,
        ("a", "t2", 0): True, ("a", "t2", 1): False,
        ("b", "t1", 0): False, ("b", "t1", 1): False,
        ("b", "t2", 0): False, ("b", "t2", 1): False,
    }
    point, low, high = paired_success_difference_interval(outcomes, first_router="a", second_router="b", draws=draws)
    assert point == 0.5
    assert (low, high) != (point, point)

