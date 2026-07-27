from __future__ import annotations

import json
from pathlib import Path

from router_benchmark.interfaces import RouteDecision, Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.webarena_live import WebArenaLive


def test_tau2_dry_adapter_applies_output_limits_and_reports_user_cost(monkeypatch) -> None:
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(cmd, cwd, capture_output, text, timeout):
        commands.append(cmd)
        save_path = Path(cmd[cmd.index("--save-to") + 1])
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "results.json").write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "reward_info": {"reward": 1.0},
                            "agent_cost": 0.12,
                            "user_cost": 0.34,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return Proc()

    monkeypatch.setattr("router_benchmark.live.tau2_live.subprocess.run", fake_run)
    monkeypatch.setattr(
        "router_benchmark.live.tau2_live.TAU2_CACHE_FILE", Path("/nonexistent/tau2_cache.json")
    )
    adapter = Tau2BenchLive.__new__(Tau2BenchLive)
    adapter.domain = "retail"
    adapter.max_steps = 7
    adapter.max_output_tokens = 123
    task = Task(
        task_id="tau2-0",
        benchmark_name=Tau2BenchLive.name,
        domain=TaskDomain.MULTI_TURN_POLICY,
        difficulty=0.0,
        requires_tool_call=True,
        candidates=LIVE_CANDIDATES,
        metadata={"tau2_task_id": 0},
    )
    decision = RouteDecision("cheap-small", 1.0, False, {})

    result = adapter.score(task, decision, rng=None)

    cmd = commands[0]
    assert cmd[cmd.index("--max-steps") + 1] == "7"
    agent_args = json.loads(cmd[cmd.index("--agent-llm-args") + 1])
    user_args = json.loads(cmd[cmd.index("--user-llm-args") + 1])
    assert agent_args["max_tokens"] == 123
    assert user_args["max_tokens"] == 123
    assert result["model_api_cost_usd"] == 0.12
    assert result["external_metered_usd"] == 0.34


def test_webarena_dry_adapter_records_nonzero_exit_without_trace(monkeypatch) -> None:
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        commands.append(cmd)
        environments.append(env)
        environments.append(env)

        class Proc:
            returncode = 1
            stdout = ""
            stderr = "login failed before model call"

        return Proc()

    monkeypatch.setattr("router_benchmark.live.webarena_live.subprocess.run", fake_run)
    adapter = WebArenaLive.__new__(WebArenaLive)
    adapter.max_steps = 7
    adapter.max_output_tokens = 123
    adapter.require_trace_cost = True
    task = Task(
        task_id="webarena-102",
        benchmark_name=WebArenaLive.name,
        domain=TaskDomain.WEB_NAVIGATION,
        difficulty=0.0,
        requires_tool_call=False,
        candidates=LIVE_CANDIDATES,
        metadata={"webarena_task_id": 102},
    )
    decision = RouteDecision("cheap-small", 1.0, False, {})

    result = adapter.score(task, decision, rng=None)

    cmd = commands[0]
    assert cmd[cmd.index("--max_steps") + 1] == "7"
    assert cmd[cmd.index("--max_tokens") + 1] == "123"
    assert json.loads(environments[0]["WEBARENA_CHROMIUM_ARGS"]) == [
        "--host-resolver-rules=MAP metis.lti.cs.cmu.edu 127.0.0.1"
    ]
    assert result["failure_status"] == "missing_provider_cost_trace"
    assert result["model_api_cost_usd"] == 0.0
