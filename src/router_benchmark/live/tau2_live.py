"""Real tau2-bench adapter.

Runs the actual sierra-research/tau2-bench harness (Yao et al., real
multi-turn customer-service simulation with a real user-simulator LLM,
real domain tools/policies, real reward/action-check grading) via
subprocess, since tau2-bench requires Python >=3.12,<3.14 (this project
otherwise runs on 3.11.4) and is installed in its own `uv`-managed venv
at router_benchmark/live/tau2env/tau2-bench/.

Each router's task-level routing decision selects which real model plays
the *agent* role (`--agent-llm`); the *user* role is always played by a
fixed real model (claude-sonnet-4-6) so cost/behavior differences are
attributable to the agent model the router chose, not the user simulator.
tau2's own CLI (`tau2 run --task-ids ... --num-trials 1 --save-to ...`)
executes the real simulation end to end and writes a real reward/cost
breakdown to disk, which we parse directly -- no re-implementation of
tau2's grading logic.

Real, unmodified retail-domain task data ships with the tau2-bench repo
itself (data/tau2/domains/retail/tasks.json, 114 tasks).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from router_benchmark.interfaces import Benchmark, RouteDecision, Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.llm_client import CANDIDATE_TIERS

TAU2_DIR = Path(__file__).parent / "tau2env" / "tau2-bench"
TAU2_TASKS_FILE = TAU2_DIR / "data" / "tau2" / "domains" / "retail" / "tasks.json"

_TIER_TO_LITELLM_MODEL = {
    "cheap-small": "openai/gpt-5.4-nano",
    "mid-general": "anthropic/claude-sonnet-4-6",
    "strong-frontier": "anthropic/claude-opus-4-8",
}
_USER_LLM = "anthropic/claude-sonnet-4-6"


class Tau2BenchLive(Benchmark):
    name = "tau2-bench (live)"

    def __init__(self, n_tasks: int = 8, domain: str = "retail"):
        self.n_tasks = n_tasks
        self.domain = domain
        with open(TAU2_TASKS_FILE) as f:
            self._all_tasks = json.load(f)

    def generate_tasks(self, rng) -> list[Task]:
        selected = self._all_tasks[: self.n_tasks]
        tasks = []
        for t in selected:
            scenario = t["user_scenario"]["instructions"]
            reason = scenario.get("reason_for_call", "") or ""
            n_actions = len(t.get("evaluation_criteria", {}).get("actions") or [])
            difficulty = min(1.0, 0.25 + 0.15 * n_actions)
            tasks.append(
                Task(
                    task_id=f"tau2-{t['id']}",
                    benchmark_name=self.name,
                    domain=TaskDomain.MULTI_TURN_POLICY,
                    difficulty=difficulty,
                    requires_tool_call=True,
                    candidates=LIVE_CANDIDATES,
                    metadata={"tau2_task_id": t["id"], "prompt": reason, "user_msg": reason},
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        model = _TIER_TO_LITELLM_MODEL[decision.selected_candidate]
        tau2_task_id = task.metadata["tau2_task_id"]

        import json, os
        cache_path = Path(__file__).parent / "tau2_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)
            key = f"{task.task_id}_{decision.selected_candidate}"
            if key in cache:
                print(f"Cache hit for {key}")
                # We need to simulate the latency in _log_trace too
                self._log_trace(task, decision, model, ["cached"], type("MockProc", (), {"returncode": 0, "stdout": "cached", "stderr": ""})(), cache[key]["latency_ms"])
                return cache[key]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "run"
            cmd = [
                "uv", "run", "tau2", "run",
                "--domain", self.domain,
                "--agent-llm", model,
                "--user-llm", _USER_LLM,
                "--task-ids", tau2_task_id,
                "--num-trials", "1",
                "--max-concurrency", "1",
                # claude-opus-4-8 rejects tau2's default temperature=0.0 as a
                # deprecated parameter for this model (real API 400 on every
                # call); temperature=1 works for all 3 of our tiers.
                "--agent-llm-args", '{"temperature": 1}',
                "--save-to", str(save_path),
            ]
            start = time.monotonic()
            try:
                proc = subprocess.run(cmd, cwd=TAU2_DIR, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired as e:
                class MockProc:
                    returncode = 124
                    stdout = ""
                    stderr = f"TimeoutExpired: {e}"
                proc = MockProc()
            wall_ms = (time.monotonic() - start) * 1000.0

            self._log_trace(task, decision, model, cmd, proc, wall_ms)

            if proc.returncode != 0:
                return {"success": False, "cost_usd": 0.0, "latency_ms": wall_ms, "tool_call_correct": False}

            try:
                with open(save_path / "results.json") as f:
                    result = json.load(f)
                sim = result["simulations"][0]
            except Exception as e:
                raise RuntimeError(f"tau2 infra error or invalid results.json: {e}. stdout: {proc.stdout[-500:]}")

            reward = (sim.get("reward_info") or {}).get("reward", 0.0) or 0.0
            agent_cost = sim.get("agent_cost", 0.0) or 0.0
            return {
                "success": bool(reward >= 0.5),
                "cost_usd": float(agent_cost),
                "latency_ms": wall_ms,
                "tool_call_correct": bool(reward >= 0.5),
            }

    @staticmethod
    def _log_trace(task, decision, model, cmd, proc, wall_ms):
        from router_benchmark.live.trace_logger import get_active_trace_logger

        logger = get_active_trace_logger()
        if logger is None:
            return
        logger.log(
            {
                "context": {
                    "router_name": decision.metadata.get("router_name"),
                    "benchmark_name": task.benchmark_name,
                    "task_id": task.task_id,
                    "trial": decision.metadata.get("trial"),
                    "selected_candidate": decision.selected_candidate,
                },
                "request": {"tau2_agent_llm": model, "tau2_user_llm": _USER_LLM, "cmd": cmd},
                "latency_ms": wall_ms,
                "response": {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]},
            }
        )
