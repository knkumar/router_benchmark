"""Real Terminal-Bench 2.0 adapter.

Uses the real `harbor` framework (laude-institute/harbor, the official
Terminal-Bench 2.0 runner) with the real `terminus-2` agent driving a real
tmux-backed terminal session inside a real Docker environment per task,
graded by each task's own real verifier (pytest-based test suite).

Task pool: an 8-task curated subset of the real terminal-bench
(laude-institute/terminal-bench) `original-tasks/` directory (241 real
tasks total), migrated to Harbor's task format via `harbor task migrate`
(router_benchmark/live/tbench_vendor/curated_harbor/). The full task set's
difficulty is wildly uneven (max_agent_timeout_sec ranges 360-1200s, and a
prior investigation timed out a "hard"-adjacent task after 17 real minutes
and $0.12); the curated subset is instead the 8 real tasks tagged
difficulty="easy" with the lowest real expert_time_estimate_min (2-5
minutes), chosen for a fixed per-phase cost/time budget rather than being
representative of the benchmark's full difficulty range -- an explicit,
documented limitation, not a hidden cherry-pick.

Each task-trial shells out to the real `harbor run` CLI (own isolated uv
tool install, not vendored) once per (router, task, trial); harbor itself
reports real cost/token usage in its own `result.json`, so no separate
cost-tracing shim is needed here (unlike webarena_live.py).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

from router_benchmark.interfaces import Benchmark, RouteDecision, Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.llm_client import CANDIDATE_TIERS, _PROVIDER_OF

TBENCH_VENDOR_DIR = Path(__file__).parent / "tbench_vendor"
CURATED_TASKS_DIR = TBENCH_VENDOR_DIR / "curated_harbor"
HARBOR_BIN = Path.home() / ".local" / "bin" / "harbor"

_DIFFICULTY_MAP = {"easy": 0.3, "medium": 0.6, "hard": 0.9}


class TerminalBenchLive(Benchmark):
    name = "Terminal-Bench 2.0 (live)"

    def __init__(self, n_tasks: int = 8):
        self.n_tasks = n_tasks
        self._task_dirs = sorted(p for p in CURATED_TASKS_DIR.iterdir() if p.is_dir())[:n_tasks]

    def generate_tasks(self, rng) -> list[Task]:
        tasks = []
        for task_dir in self._task_dirs:
            with open(task_dir / "task.toml", "rb") as f:
                meta = tomllib.load(f)["metadata"]
            instruction = (task_dir / "instruction.md").read_text()
            difficulty = _DIFFICULTY_MAP.get(meta.get("difficulty"), 0.5)
            tasks.append(
                Task(
                    task_id=f"tbench-{task_dir.name}",
                    benchmark_name=self.name,
                    domain=TaskDomain.TERMINAL_AGENT,
                    difficulty=difficulty,
                    requires_tool_call=True,
                    candidates=LIVE_CANDIDATES,
                    metadata={
                        "task_dir": str(task_dir),
                        "task_name": task_dir.name,
                        "prompt": instruction,
                        "user_msg": instruction,
                        "category": meta.get("category"),
                    },
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        model = CANDIDATE_TIERS[decision.selected_candidate]
        provider = _PROVIDER_OF[model]
        task_dir = task.metadata["task_dir"]

        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir) / "jobs"
            cmd = [
                str(HARBOR_BIN), "run",
                "--path", task_dir,
                "-a", "terminus-2",
                "-m", f"{provider}/{model}",
                "--jobs-dir", str(jobs_dir),
                "-y", "-n", "1",
            ]
            start = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                timed_out = False
            except subprocess.TimeoutExpired as e:
                proc = e
                timed_out = True
            wall_ms = (time.monotonic() - start) * 1000.0

            success, cost = False, 0.0
            if not timed_out:
                result_files = list(jobs_dir.glob("*/result.json"))
                if result_files:
                    result = json.loads(result_files[0].read_text())
                    evals = result.get("stats", {}).get("evals", {})
                    for ev in evals.values():
                        metrics = ev.get("metrics") or []
                        if metrics and metrics[0].get("mean") is not None:
                            success = bool(metrics[0]["mean"] >= 1.0)
                    cost = result.get("stats", {}).get("cost_usd") or 0.0

            self._log_trace(task, decision, model, provider, cmd, proc, wall_ms, timed_out, success)

            return {
                "success": success,
                "cost_usd": cost,
                "latency_ms": wall_ms,
                "tool_call_correct": success if task.requires_tool_call else None,
            }

    @staticmethod
    def _log_trace(task, decision, model, provider, cmd, proc, wall_ms, timed_out, success):
        from router_benchmark.live.trace_logger import get_active_trace_logger

        logger = get_active_trace_logger()
        if logger is None:
            return
        record = {
            "context": {
                "router_name": decision.metadata.get("router_name"),
                "benchmark_name": task.benchmark_name,
                "task_id": task.task_id,
                "trial": decision.metadata.get("trial"),
                "selected_candidate": decision.selected_candidate,
                "success": success,
            },
            "request": {"provider": provider, "model": model, "cmd": cmd},
            "latency_ms": wall_ms,
        }
        if timed_out:
            record["error"] = "TimeoutExpired"
        else:
            record["response"] = {
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        logger.log(record)
