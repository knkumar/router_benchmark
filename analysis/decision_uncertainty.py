"""Rank and Pareto uncertainty summaries from saved resampling draws."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from router_benchmark.protocol.pareto import pareto_membership_with_witness


def average_tie_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Competition ranks for higher-is-better values with averaged ties."""
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2
        for router, _ in ordered[index:end]:
            ranks[router] = rank
        index = end
    return ranks


def rank_uncertainty(draws: Sequence[Mapping[str, float]]) -> dict[str, dict[str, object]]:
    """Rank probabilities, mean rank, variance, and pairwise inversions."""
    if not draws:
        raise ValueError("draws must be nonempty")
    routers = set(draws[0])
    if not routers or any(set(draw) != routers for draw in draws):
        raise ValueError("every draw must cover the same nonempty routers")
    per_router: dict[str, list[float]] = {router: [] for router in routers}
    inversions: Counter[tuple[str, str]] = Counter()
    for draw in draws:
        ranks = average_tie_ranks(draw)
        for router, rank in ranks.items():
            per_router[router].append(rank)
        for first in routers:
            for second in routers:
                if first != second and ranks[first] > ranks[second]:
                    inversions[(first, second)] += 1
    result: dict[str, dict[str, object]] = {}
    for router, ranks in per_router.items():
        probabilities = {rank: ranks.count(rank) / len(ranks) for rank in sorted(set(ranks))}
        mean = sum(ranks) / len(ranks)
        variance = sum((rank - mean) ** 2 for rank in ranks) / len(ranks)
        result[router] = {
            "rank_probabilities": probabilities,
            "mean_rank": mean,
            "rank_variance": variance,
            "pairwise_inversion_probability": {
                other: inversions[(router, other)] / len(draws) for other in sorted(routers - {router})
            },
        }
    return result


def pareto_nondominance_probabilities(
    draws: Sequence[Sequence[Mapping[str, object]]], *, id_key: str, cost_key: str, success_key: str
) -> dict[str, float]:
    """Fraction of saved draws in which each point is nondominated."""
    if not draws:
        raise ValueError("draws must be nonempty")
    counts: Counter[str] = Counter()
    expected_ids: set[str] | None = None
    for draw in draws:
        membership = pareto_membership_with_witness(draw, id_key=id_key, cost_key=cost_key, success_key=success_key)
        ids = {str(row["point_id"]) for row in membership}
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("every Pareto draw must cover the same points")
        for row in membership:
            if row["is_pareto_nondominated"]:
                counts[str(row["point_id"])] += 1
    return {identifier: counts[identifier] / len(draws) for identifier in sorted(expected_ids or set())}
