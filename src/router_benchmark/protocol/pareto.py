"""Validated Pareto-frontier membership for cost-versus-success points."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


class ParetoValidationError(ValueError):
    """Raised when a point cannot be compared on the Pareto frontier."""


def pareto_membership_with_witness(
    points: Iterable[Mapping[str, Any]],
    *,
    id_key: str,
    cost_key: str,
    success_key: str,
) -> list[dict[str, Any]]:
    """Return Pareto membership and one dominating point, when present.

    Lower cost and higher success are preferred. Every identifier must be
    unique, and cost and success must be finite numeric values.
    """
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for point in points:
        try:
            identifier = str(point[id_key])
        except KeyError as error:
            raise ParetoValidationError(f"points must contain {id_key}") from error
        if not identifier or identifier in identifiers:
            raise ParetoValidationError("point identifiers must be unique and nonempty")
        identifiers.add(identifier)
        try:
            cost = float(point[cost_key])
            success = float(point[success_key])
        except (KeyError, TypeError, ValueError) as error:
            raise ParetoValidationError("points must contain numeric cost and success") from error
        if not math.isfinite(cost) or not math.isfinite(success):
            raise ParetoValidationError(f"non-finite metric for {identifier}")
        if cost < 0 or not 0 <= success <= 1:
            raise ParetoValidationError(f"out-of-bounds metric for {identifier}")
        normalized.append({"id": identifier, "cost": cost, "success": success})

    results: list[dict[str, Any]] = []
    for point in normalized:
        witness = next(
            (
                other["id"]
                for other in normalized
                if other["id"] != point["id"]
                and other["cost"] <= point["cost"]
                and other["success"] >= point["success"]
                and (other["cost"] < point["cost"] or other["success"] > point["success"])
            ),
            None,
        )
        results.append(
            {
                "point_id": point["id"],
                "is_pareto_nondominated": witness is None,
                "dominated_by": witness,
            }
        )
    return results
