"""Simple content-free baseline routers, evaluated through the same
EvaluationHarness every real router adapter goes through -- these answer
the review's core question of whether the real routers beat trivial
policies, not just each other."""

from __future__ import annotations

from router_benchmark.interfaces import RouteDecision, Router, Task


class AlwaysCheapestRouter(Router):
    name = "Always-Cheapest Baseline (live)"

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        return RouteDecision(selected_candidate="cheap-small", confidence=1.0, fallback_used=False)


class AlwaysStrongestRouter(Router):
    name = "Always-Strongest Baseline (live)"

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        return RouteDecision(selected_candidate="strong-frontier", confidence=1.0, fallback_used=False)


class RandomRouter(Router):
    name = "Random Baseline (live)"
    _TIERS = ("cheap-small", "mid-general", "strong-frontier")

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        tier = self._TIERS[rng.integers(0, len(self._TIERS))]
        return RouteDecision(selected_candidate=tier, confidence=1.0, fallback_used=False)


class PromptLengthHeuristicRouter(Router):
    """Routes on raw prompt length only: short -> cheap, medium -> mid,
    long -> strong. Thresholds are word counts, not content-aware."""

    name = "Prompt-Length Heuristic Baseline (live)"
    _SHORT_MAX_WORDS = 40
    _MEDIUM_MAX_WORDS = 150

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        n_words = len(prompt.split())
        if n_words <= self._SHORT_MAX_WORDS:
            tier = "cheap-small"
        elif n_words <= self._MEDIUM_MAX_WORDS:
            tier = "mid-general"
        else:
            tier = "strong-frontier"
        return RouteDecision(selected_candidate=tier, confidence=1.0, fallback_used=False, metadata={"n_words": n_words})


class ToolRequiredHeuristicRouter(Router):
    """Routes cheap-small unless the task requires a tool call, in which
    case it escalates to mid-general -- a common real-world heuristic
    that tool-calling correctness needs a stronger model."""

    name = "Tool-Required Heuristic Baseline (live)"

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        tier = "mid-general" if task.requires_tool_call else "cheap-small"
        return RouteDecision(selected_candidate=tier, confidence=1.0, fallback_used=False)


def build_all_baselines() -> list:
    return [
        AlwaysCheapestRouter(),
        AlwaysStrongestRouter(),
        RandomRouter(),
        PromptLengthHeuristicRouter(),
        ToolRequiredHeuristicRouter(),
    ]
