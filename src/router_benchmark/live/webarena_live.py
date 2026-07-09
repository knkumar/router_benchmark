"""Real WebArena adapter.

Runs the actual web-arena-x/webarena harness (Zhou et al.) via subprocess,
since WebArena requires an old, pinned dependency stack (openai==0.27.0,
playwright==1.32.1, transformers==4.33.2) incompatible with this project's
main environment, and is installed in its own venv at
~/.local/share/router_bench_vendor/webarena/.venv (cloned from
github.com/web-arena-x/webarena, not vendored into this repo -- kept outside
/tmp specifically because /tmp on this shared host gets wiped periodically
by a co-tenant process).

Two vendored patches were applied directly to that clone (not upstream,
since WebArena's own LLM call path only supported "openai"/"huggingface"):
  - llms/providers/anthropic_utils.py (new file): a minimal Anthropic chat
    completion call mirroring generate_from_openai_chat_completion's
    signature, so our mid-general/strong-frontier (Claude) tiers can back
    the PromptAgent's calls, not just cheap-small (OpenAI).
  - llms/providers/openai_utils.py: gpt-5.x-era models reject the legacy
    "max_tokens" kwarg this openai==0.27.0-era call site was written
    against (same class of bug already hit and fixed for LiteLLM Router
    against tau2-bench); patched to send max_completion_tokens for gpt-5*.
  - Both provider call sites now append {model, input_tokens, output_tokens}
    JSONL lines to $WEBARENA_TRACE_FILE per real call, since WebArena's own
    harness does not track cost -- this is the only way we recover real
    token usage for cost accounting.

Each router's routing decision selects which real model plays the agent
in WebArena's own ScriptBrowserEnv + PromptAgent + evaluator_router loop
(ReAct-style: accessibility-tree observation -> action -> real Playwright
step against a self-hosted site -> repeat until stop or max_steps). We
shell out to the real run.py per task (one task per subprocess, so the
router's per-task model selection can vary) and parse the real
"Average score: <float>" line run.py logs at the end of a single-task
run -- WebArena's own evaluator_router output, not a re-implementation.

Task pool: real task configs from config_files/test.raw.json (812 total),
filtered to single-site "gitlab" or "shopping" tasks only, since those are
the two sites actually self-hosted for this project (reddit/wikipedia/map/
shopping_admin were not stood up).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from router_benchmark.interfaces import Benchmark, RouteDecision, Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.llm_client import CANDIDATE_TIERS, PRICING, _PROVIDER_OF

WEBARENA_DIR = Path.home() / ".local" / "share" / "router_bench_vendor" / "webarena"
WEBARENA_PYTHON = WEBARENA_DIR / ".venv" / "bin" / "python"
CONFIG_FILES_DIR = WEBARENA_DIR / "config_files"
RAW_CONFIG_FILE = CONFIG_FILES_DIR / "test.raw.json"

_SITE_ENV = {
    "REDDIT": "http://localhost:9999",
    "SHOPPING": "http://localhost:7770",
    "SHOPPING_ADMIN": "http://localhost:7780/admin",
    "GITLAB": "http://localhost:8023",
    "WIKIPEDIA": "http://localhost:8888",
    "MAP": "http://localhost:3000",
    "HOMEPAGE": "http://localhost:4399",
}

_SCORE_RE = re.compile(r"Average score:\s*([0-9.]+)")


class WebArenaLive(Benchmark):
    name = "WebArena (live)"

    def __init__(self, n_tasks: int = 8, sites: tuple[str, ...] = ("gitlab", "shopping")):
        self.n_tasks = n_tasks
        with open(RAW_CONFIG_FILE) as f:
            all_tasks = json.load(f)
        self._pool = [t for t in all_tasks if tuple(t["sites"]) in [(s,) for s in sites]]

    def generate_tasks(self, rng) -> list[Task]:
        selected = self._pool[: self.n_tasks]
        tasks = []
        for t in selected:
            n_eval_types = len(t.get("eval", {}).get("eval_types", []))
            intent_len = len(t.get("intent", ""))
            difficulty = min(1.0, 0.3 + 0.1 * n_eval_types + 0.001 * intent_len)
            tasks.append(
                Task(
                    task_id=f"webarena-{t['task_id']}",
                    benchmark_name=self.name,
                    domain=TaskDomain.WEB_NAVIGATION,
                    difficulty=difficulty,
                    requires_tool_call=False,
                    candidates=LIVE_CANDIDATES,
                    metadata={
                        "webarena_task_id": t["task_id"],
                        "sites": t["sites"],
                        "intent": t["intent"],
                        # content-based routers (Aurelio, RouteLLM, NVIDIA
                        # Blueprint, LLMRouter, vLLM Semantic Router) key off
                        # metadata["prompt"]/["user_msg"]; without this they
                        # silently fall back to a fixed default for every
                        # WebArena task instead of actually routing on intent.
                        "prompt": t["intent"],
                        "user_msg": t["intent"],
                    },
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        model = CANDIDATE_TIERS[decision.selected_candidate]
        provider = _PROVIDER_OF[model]
        task_id = task.metadata["webarena_task_id"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "result"
            trace_file = Path(tmpdir) / "trace.jsonl"
            cmd = [
                str(WEBARENA_PYTHON), "run.py",
                "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
                "--test_start_idx", str(task_id),
                "--test_end_idx", str(task_id + 1),
                "--provider", provider,
                "--model", model,
                "--result_dir", str(result_dir),
                # claude-opus-4-8 rejects temperature=0.0 as deprecated for
                # this model (same bug already hit/fixed for tau2-bench);
                # temperature=1.0 works for all 3 of our tiers.
                "--temperature", "1.0",
                "--max_tokens", "384",
            ]
            env = dict(os.environ)
            env.update(_SITE_ENV)
            env["WEBARENA_TRACE_FILE"] = str(trace_file)
            env["PYTHONPATH"] = str(WEBARENA_DIR)
            # run.py shells out to "python browser_env/auto_login.py" using
            # a bare "python" from PATH; without this it resolves to the
            # system/pyenv python, not our venv with playwright installed.
            env["PATH"] = f"{WEBARENA_DIR / '.venv' / 'bin'}:{env.get('PATH', '')}"

            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, cwd=WEBARENA_DIR, env=env,
                    capture_output=True, text=True, timeout=900,
                )
                timed_out = False
            except subprocess.TimeoutExpired as e:
                proc = e
                timed_out = True
            wall_ms = (time.monotonic() - start) * 1000.0

            cost = self._read_trace_cost(trace_file)
            self._log_trace(task, decision, model, provider, cmd, proc, wall_ms, timed_out)

            if timed_out:
                return {"success": False, "cost_usd": cost, "latency_ms": wall_ms, "tool_call_correct": None}

            combined = (proc.stdout or "") + (proc.stderr or "")
            m = _SCORE_RE.search(combined)
            score = float(m.group(1)) if m else 0.0
            return {"success": bool(score >= 1.0), "cost_usd": cost, "latency_ms": wall_ms, "tool_call_correct": None}

    @staticmethod
    def _read_trace_cost(trace_file: Path) -> float:
        if not trace_file.exists():
            return 0.0
        total = 0.0
        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                price_in, price_out = PRICING[rec["model"]]
                total += rec["input_tokens"] * price_in + rec["output_tokens"] * price_out
        return total

    @staticmethod
    def _log_trace(task, decision, model, provider, cmd, proc, wall_ms, timed_out):
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
            },
            "request": {"webarena_provider": provider, "webarena_model": model, "cmd": cmd},
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
