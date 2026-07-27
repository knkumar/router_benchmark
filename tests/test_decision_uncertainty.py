from __future__ import annotations

from router_benchmark.analysis.decision_uncertainty import pareto_nondominance_probabilities, rank_uncertainty


def test_rank_probabilities_sum_to_one_and_capture_ties() -> None:
    result = rank_uncertainty([
        {"a": 0.9, "b": 0.8, "c": 0.7},
        {"a": 0.8, "b": 0.8, "c": 0.7},
    ])
    assert all(sum(summary["rank_probabilities"].values()) == 1 for summary in result.values())
    assert result["a"]["rank_probabilities"] == {1.0: 0.5, 1.5: 0.5}


def test_pareto_probability_equals_fraction_of_nondominated_draws() -> None:
    probability = pareto_nondominance_probabilities([
        [{"id": "a", "cost": 1.0, "success": 0.8}, {"id": "b", "cost": 2.0, "success": 0.7}],
        [{"id": "a", "cost": 1.0, "success": 0.6}, {"id": "b", "cost": 2.0, "success": 0.7}],
    ], id_key="id", cost_key="cost", success_key="success")
    assert probability == {"a": 1.0, "b": 0.5}

