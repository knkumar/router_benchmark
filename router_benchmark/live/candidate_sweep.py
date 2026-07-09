"""ForcedTierRouter: always selects one fixed tier, regardless of task
content -- used only to sweep every candidate tier's real outcome per
task (oracle upper bound, pessimal bound, regret-to-oracle, normalized
cost), not as a router under comparison."""

from __future__ import annotations

from router_benchmark.interfaces import RouteDecision, Router, Task


class ForcedTierRouter(Router):
    def __init__(self, tier: str):
        assert tier in ("cheap-small", "mid-general", "strong-frontier")
        self.tier = tier
        self.name = f"__sweep__{tier}"

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        return RouteDecision(selected_candidate=self.tier, confidence=1.0, fallback_used=False)
