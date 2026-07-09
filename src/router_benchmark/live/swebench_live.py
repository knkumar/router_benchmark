"""Real SWE-bench Verified adapter.

Uses the actual princeton-nlp/SWE-bench_Verified dataset (500 real,
human-verified GitHub issues, Jimenez et al.) and the real official
`swebench` grading harness (swebench.harness.run_evaluation), which pulls
real per-instance Docker images and runs the real repository's real test
suite against whatever patch we submit.

LIMITATION: the router's selected model generates a patch zero-shot from
just the issue's `problem_statement` text -- no repository browsing, no
tool use, no iterative debugging. This is a much weaker baseline than a
real coding agent (which would explore the repo, run tests, and iterate),
but it is a real, live-executed, honestly-labeled data point: the model
genuinely attempts the real fix and is graded by the real test suite, not
simulated. See router_benchmark/live/README.md.

Each task-trial invokes the real grading harness as a subprocess (own
Python 3.11 env is fine here; swebench itself, unlike tau2-bench, doesn't
require Python >=3.12) since each real evaluation takes several minutes
of real Docker container time for the repository's real test suite.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from datasets import load_dataset

from router_benchmark.interfaces import Benchmark, RouteDecision, Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.llm_client import CANDIDATE_TIERS, LiveLLMClient

SWEBENCH_WORKDIR = Path(__file__).parent / "swebench_env"

PATCH_SYSTEM_PROMPT = """You are an expert software engineer. You will be given a real GitHub issue.
Produce a single unified diff (git diff format, starting with `--- a/...` / `+++ b/...` and `@@` hunks)
that fixes the issue. Output ONLY the diff, no explanation, no markdown code fences."""


class SWEBenchLive(Benchmark):
    name = "SWE-bench Verified (live)"

    def __init__(self, instance_ids: list[str] | None = None, n_tasks: int = 2):
        self.instance_ids = instance_ids
        self.n_tasks = n_tasks
        self._client = LiveLLMClient()
        self._rows_by_id = None

    def _load(self):
        if self._rows_by_id is None:
            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            if self.instance_ids:
                rows = [r for r in ds if r["instance_id"] in self.instance_ids]
            else:
                rows = [r for r in ds if r["difficulty"] == "<15 min fix"][: self.n_tasks]
            self._rows_by_id = {r["instance_id"]: r for r in rows}
        return self._rows_by_id

    def generate_tasks(self, rng) -> list[Task]:
        rows = self._load()
        tasks = []
        for iid, row in rows.items():
            difficulty = {"<15 min fix": 0.3, "15 min - 1 hour": 0.5, "1-4 hours": 0.75, ">4 hours": 0.95}.get(
                row["difficulty"], 0.5
            )
            tasks.append(
                Task(
                    task_id=f"swebench-{iid}",
                    benchmark_name=self.name,
                    domain=TaskDomain.CODE_REPAIR,
                    difficulty=difficulty,
                    requires_tool_call=False,
                    candidates=LIVE_CANDIDATES,
                    metadata={"instance_id": iid, "prompt": row["problem_statement"], "repo": row["repo"]},
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        model = CANDIDATE_TIERS[decision.selected_candidate]
        instance_id = task.metadata["instance_id"]

        gen = self._client.call(
            model=model,
            system=PATCH_SYSTEM_PROMPT,
            user=f"Repository: {task.metadata['repo']}\n\nIssue:\n{task.metadata['prompt']}",
            max_tokens=2000,
            trace_context={
                "router_name": decision.metadata.get("router_name"),
                "benchmark_name": task.benchmark_name,
                "task_id": task.task_id,
                "trial": decision.metadata.get("trial"),
                "selected_candidate": decision.selected_candidate,
                "role": "swebench_patch_generation",
            },
        )
        patch = gen.text.strip()
        if patch.startswith("```"):
            patch = "\n".join(line for line in patch.splitlines() if not line.strip().startswith("```"))

        router_tag = (decision.metadata.get("router_name") or "router").replace(" ", "_").replace("(", "").replace(")", "")
        run_id = f"{router_tag}-t{decision.metadata.get('trial', 0)}"
        model_name_tag = f"{router_tag}__{decision.selected_candidate}"

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = Path(tmpdir) / "predictions.json"
            with open(pred_path, "w") as f:
                json.dump([{"instance_id": instance_id, "model_patch": patch, "model_name_or_path": model_name_tag}], f)

            cmd = [
                # bare "python3" resolves via PATH to whatever interpreter the
                # calling process inherits, not necessarily the venv with
                # swebench installed -- pin explicitly to avoid a silent
                # ModuleNotFoundError before any Docker evaluation runs.
                str(Path(__file__).parent.parent / ".venv" / "bin" / "python3"),
                "-m", "swebench.harness.run_evaluation",
                "-d", "princeton-nlp/SWE-bench_Verified",
                "-p", str(pred_path),
                "--instance_ids", instance_id,
                "-id", run_id,
                "--max_workers", "1",
                "--cache_level", "instance",
            ]
            start = time.monotonic()
            proc = subprocess.run(cmd, cwd=SWEBENCH_WORKDIR, capture_output=True, text=True, timeout=1800)
            wall_ms = (time.monotonic() - start) * 1000.0 + gen.latency_ms

            resolved = False
            report_path = SWEBENCH_WORKDIR / "logs" / "run_evaluation" / run_id / model_name_tag / instance_id / "report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                resolved = bool(report.get(instance_id, {}).get("resolved", False))

            self._log_trace(task, decision, model_name_tag, cmd, proc, resolved)

        return {
            "success": resolved,
            "cost_usd": gen.cost_usd,
            "latency_ms": wall_ms,
            "tool_call_correct": None,
        }

    @staticmethod
    def _log_trace(task, decision, model_name_tag, cmd, proc, resolved):
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
                    "resolved": resolved,
                },
                "request": {"swebench_cmd": cmd, "model_name_tag": model_name_tag},
                "response": {"returncode": proc.returncode, "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]},
            }
        )


def build_swebench_live(n_tasks: int = 2) -> SWEBenchLive:
    return SWEBenchLive(instance_ids=["astropy__astropy-14309", "astropy__astropy-14995"][:n_tasks])
