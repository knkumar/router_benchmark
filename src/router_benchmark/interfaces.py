"""Common interface that every router and every benchmark is normalized behind.

This mirrors the "Recommended Comparative Evaluation Protocol" in
agentic_routing_router_benchmark_shortlist.md:

    route(input, context, candidates) -> selected_candidate, confidence, metadata

A benchmark is a source of Task objects plus an oracle that scores a route
decision. A router is an object that, given a task and a candidate pool,
returns a RouteDecision. The EvaluationHarness (harness.py) is the only
place that wires the two together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskDomain(str, Enum):
    QA_REASONING = "qa_reasoning"
    CODE_REPAIR = "code_repair"
    TOOL_USE = "tool_use"
    MULTI_TURN_POLICY = "multi_turn_policy"
    WEB_NAVIGATION = "web_navigation"
    TERMINAL_AGENT = "terminal_agent"


@dataclass(frozen=True)
class Candidate:
    """One routable option: a model, tool, or agent policy."""

    name: str
    tier: str  # "cheap", "mid", "strong"
    cost_per_1k_tokens: float
    base_quality: float  # 0-1, candidate's raw capability on a hard task
    base_latency_ms: float


@dataclass(frozen=True)
class Task:
    """A single benchmark task instance."""

    task_id: str
    benchmark_name: str
    domain: TaskDomain
    difficulty: float  # 0-1, higher = harder
    requires_tool_call: bool
    candidates: tuple[Candidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    """What a router chose for a given task."""

    selected_candidate: str
    confidence: float  # 0-1
    fallback_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Outcome of executing one task through one router, one trial."""

    router_name: str
    benchmark_name: str
    task_id: str
    domain: TaskDomain
    difficulty: float
    trial: int
    selected_candidate: str
    confidence: float
    fallback_used: bool
    success: bool
    tool_call_required: bool
    tool_call_correct: bool | None  # None if no tool call required
    cost_usd: float
    latency_ms: float
    router_decision_latency_ms: float = 0.0  # wall-clock time inside router.route() only, excluding downstream generation


class Router(ABC):
    """Adapter interface every router candidate must implement.

    Concrete adapters in routers.py wrap the six shortlisted open-source
    routers. Because this environment has no network access or model-provider
    credentials, each adapter's `route()` is backed by a documented, seeded
    synthetic behavior profile rather than a live call into the upstream
    project. See routers.py module docstring for details and citations.
    """

    name: str

    @abstractmethod
    def route(self, task: Task, context: dict[str, Any], rng: Any) -> RouteDecision:
        """Select a candidate for `task` from `task.candidates`."""
        raise NotImplementedError


class Benchmark(ABC):
    """Adapter interface every benchmark must implement."""

    name: str
    # True when score() is deterministic and safe to evaluate once per
    # (task, candidate, fallback) for all router comparisons. Live benchmarks
    # that make real model calls must set this False.
    reusable_score: bool = True

    @abstractmethod
    def generate_tasks(self, rng: Any) -> list[Task]:
        """Produce the fixed task set for this benchmark."""
        raise NotImplementedError

    @abstractmethod
    def score(self, task: Task, decision: RouteDecision, rng: Any) -> dict[str, Any]:
        """Execute/score a route decision against this benchmark's oracle.

        Returns a dict with at least: success (bool), cost_usd (float),
        latency_ms (float), and, if task.requires_tool_call, tool_call_correct
        (bool).
        """
        raise NotImplementedError
