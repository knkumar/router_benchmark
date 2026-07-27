from __future__ import annotations

import pytest

from router_benchmark.protocol.pareto import ParetoValidationError, pareto_membership_with_witness


def test_pareto_emits_dominance_witness_from_unrounded_values() -> None:
    results = pareto_membership_with_witness(
        [
            {"router": "cheap", "cost": 0.1, "success": 0.6},
            {"router": "strong", "cost": 1.0, "success": 0.9},
            {"router": "dominated", "cost": 1.1, "success": 0.8},
        ], id_key="router", cost_key="cost", success_key="success"
    )
    by_router = {row["point_id"]: row for row in results}
    assert by_router["cheap"]["is_pareto_nondominated"] is True
    assert by_router["strong"]["is_pareto_nondominated"] is True
    assert by_router["dominated"] == {
        "point_id": "dominated", "is_pareto_nondominated": False, "dominated_by": "strong"
    }


def test_pareto_rejects_negative_cost() -> None:
    with pytest.raises(ParetoValidationError, match="out-of-bounds"):
        pareto_membership_with_witness(
            [{"router": "bad", "cost": -0.01, "success": 0.5}],
            id_key="router", cost_key="cost", success_key="success"
        )

