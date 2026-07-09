"""Benchmark adapters for the six shortlisted suites.

SIMULATED BACKEND: as with routers.py, there is no network access to fetch
real RouterBench outcome tables, run SWE-bench Verified containers, execute
BFCL function-calling harnesses, run tau2-bench multi-turn sessions, drive
WebArena browsers, or run Terminal-Bench 2.0 tasks in this environment. Each
benchmark below is a seeded synthetic task generator whose task-count,
difficulty distribution, tool-call fraction, domain, token-cost scale, and
latency scale are set to be directionally consistent with the benchmark's
published design (cited per class). The scoring function
(`Benchmark.score`) is shared logic in `_score_common` that turns a router's
candidate choice + task difficulty into a stochastic success/cost/latency/
tool-accuracy outcome — this is the "oracle" a live harness would otherwise
get by actually executing the task.

Citations (see paper/paper.md references for full entries):
  RouterBench      -- Hu et al., 2024 [7]
  SWE-bench Verified -- OpenAI / Jimenez et al. [8]
  BFCL v4           -- Patil et al., Gorilla project [9]
  tau-bench/tau2    -- Yao et al., Sierra Research [10]
  WebArena          -- Zhou et al. [11]
  Terminal-Bench 2.0 -- Terminal-Bench / Harbor [12]
"""

from __future__ import annotations

import numpy as np

from router_benchmark.interfaces import Benchmark, Candidate, RouteDecision, Task, TaskDomain

# Shared candidate pool: cheap / mid / strong, roughly matching the
# cost-quality spread described across the shortlisted routers' docs.
CANDIDATE_POOL: tuple[Candidate, ...] = (
    Candidate("cheap-small", tier="cheap", cost_per_1k_tokens=0.0002, base_quality=0.55, base_latency_ms=180),
    Candidate("mid-general", tier="mid", cost_per_1k_tokens=0.0020, base_quality=0.75, base_latency_ms=420),
    Candidate("strong-frontier", tier="strong", cost_per_1k_tokens=0.0150, base_quality=0.93, base_latency_ms=950),
)


def _score_common(
    task: Task,
    decision: RouteDecision,
    rng: np.random.Generator,
    tokens_per_task: float,
    latency_jitter: float,
) -> dict:
    candidate = next(c for c in task.candidates if c.name == decision.selected_candidate)

    # Logistic success model: candidate quality vs. task difficulty.
    z = 6.0 * (candidate.base_quality - (0.3 + 0.65 * task.difficulty))
    p_success = 1.0 / (1.0 + np.exp(-z))
    success = bool(rng.random() < p_success)

    cost_usd = candidate.cost_per_1k_tokens * (tokens_per_task / 1000.0) * rng.uniform(0.85, 1.20)
    latency_ms = candidate.base_latency_ms * (1.0 + 0.6 * task.difficulty)
    latency_ms *= rng.uniform(1 - latency_jitter, 1 + latency_jitter)
    if decision.fallback_used:
        latency_ms *= 1.6
        cost_usd *= 1.4

    tool_call_correct = None
    if task.requires_tool_call:
        p_tool = np.clip(p_success * 0.95 + 0.05, 0.02, 0.99)
        tool_call_correct = bool(rng.random() < p_tool)
        if not tool_call_correct:
            success = False

    return {
        "success": success,
        "cost_usd": float(cost_usd),
        "latency_ms": float(latency_ms),
        "tool_call_correct": tool_call_correct,
    }


class _SyntheticBenchmark(Benchmark):
    def __init__(
        self,
        name: str,
        domain: TaskDomain,
        n_tasks: int,
        difficulty_range: tuple[float, float],
        tool_fraction: float,
        tokens_per_task: float,
        latency_jitter: float,
    ):
        self.name = name
        self.domain = domain
        self.n_tasks = n_tasks
        self.difficulty_range = difficulty_range
        self.tool_fraction = tool_fraction
        self.tokens_per_task = tokens_per_task
        self.latency_jitter = latency_jitter

    def generate_tasks(self, rng: np.random.Generator) -> list[Task]:
        lo, hi = self.difficulty_range
        tasks = []
        for i in range(self.n_tasks):
            difficulty = float(np.clip(rng.beta(2, 2) * (hi - lo) + lo, 0.0, 1.0))
            requires_tool = bool(rng.random() < self.tool_fraction)
            tasks.append(
                Task(
                    task_id=f"{self.name.replace(' ', '_')}-{i:04d}",
                    benchmark_name=self.name,
                    domain=self.domain,
                    difficulty=difficulty,
                    requires_tool_call=requires_tool,
                    candidates=CANDIDATE_POOL,
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng: np.random.Generator) -> dict:
        return _score_common(task, decision, rng, self.tokens_per_task, self.latency_jitter)


def make_routerbench() -> _SyntheticBenchmark:
    """RouterBench [7]: >405k logged multi-LLM outcomes; the canonical
    router-only, single-turn benchmark. Short, cheap, low-latency-jitter
    single-shot QA/reasoning tasks with a wide difficulty spread."""
    return _SyntheticBenchmark(
        name="RouterBench",
        domain=TaskDomain.QA_REASONING,
        n_tasks=400,
        difficulty_range=(0.05, 0.95),
        tool_fraction=0.0,
        tokens_per_task=600,
        latency_jitter=0.15,
    )


def make_swebench_verified() -> _SyntheticBenchmark:
    """SWE-bench Verified [8]: real GitHub issue resolution. Higher token
    volume (repo context + patch), skewed toward harder difficulty, no
    explicit tool-call scoring axis (tests pass/fail is the oracle) but
    long-horizon so latency/cost scale up sharply."""
    return _SyntheticBenchmark(
        name="SWE-bench Verified",
        domain=TaskDomain.CODE_REPAIR,
        n_tasks=200,
        difficulty_range=(0.3, 1.0),
        tool_fraction=0.0,
        tokens_per_task=6000,
        latency_jitter=0.35,
    )


def make_bfcl_v4() -> _SyntheticBenchmark:
    """BFCL v4 [9]: function/tool-calling correctness, including executable
    and agentic scenarios. Every task requires a tool call; the benchmark's
    entire purpose is measuring tool-call accuracy under routing."""
    return _SyntheticBenchmark(
        name="BFCL v4",
        domain=TaskDomain.TOOL_USE,
        n_tasks=300,
        difficulty_range=(0.1, 0.9),
        tool_fraction=1.0,
        tokens_per_task=900,
        latency_jitter=0.20,
    )


def make_tau_bench() -> _SyntheticBenchmark:
    """tau-bench/tau2-bench [10]: multi-turn customer-service agents with
    domain APIs and policies. High tool-call fraction (API calls each
    turn), moderate-high difficulty, moderate token volume for multi-turn
    context."""
    return _SyntheticBenchmark(
        name="tau2-bench",
        domain=TaskDomain.MULTI_TURN_POLICY,
        n_tasks=150,
        difficulty_range=(0.25, 0.9),
        tool_fraction=0.8,
        tokens_per_task=2500,
        latency_jitter=0.30,
    )


def make_webarena() -> _SyntheticBenchmark:
    """WebArena [11]: realistic web-navigation agents on self-hosted sites.
    Long horizon (many browser steps), high latency jitter (network/DOM
    variance), moderate tool-call fraction (structured actions)."""
    return _SyntheticBenchmark(
        name="WebArena",
        domain=TaskDomain.WEB_NAVIGATION,
        n_tasks=180,
        difficulty_range=(0.3, 0.95),
        tool_fraction=0.6,
        tokens_per_task=4000,
        latency_jitter=0.45,
    )


def make_terminal_bench() -> _SyntheticBenchmark:
    """Terminal-Bench 2.0 [12]: terminal agents on realistic end-to-end
    tasks in containerized environments. Longest horizon of the six,
    heavy tool-call fraction (shell commands), highest cost/latency
    scale, reflecting its role as the hardest long-horizon routing test."""
    return _SyntheticBenchmark(
        name="Terminal-Bench 2.0",
        domain=TaskDomain.TERMINAL_AGENT,
        n_tasks=150,
        difficulty_range=(0.35, 1.0),
        tool_fraction=0.9,
        tokens_per_task=7000,
        latency_jitter=0.40,
    )


def build_all_benchmarks() -> list[Benchmark]:
    """Convenience factory used by run.py."""
    return [
        make_routerbench(),
        make_swebench_verified(),
        make_bfcl_v4(),
        make_tau_bench(),
        make_webarena(),
        make_terminal_bench(),
    ]
