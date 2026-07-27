# router_benchmark/tests/test_routing_proxy_e2e_offline.py
import json
import unittest
from fastapi.testclient import TestClient
from router_benchmark.interfaces import RouteDecision
from router_benchmark.live.routing_context import PerRequestRouter, latest_routing_text
from router_benchmark.live.routing_proxy import build_proxy_app
from router_benchmark.live.tau2_live import _rollup_cost_from_steps, _escalation_count


class _HardWordRouter(PerRequestRouter):
    """Cheap by default; escalates to strong when the latest turn says 'hard'."""
    name = "hardword"
    def _route_on_text(self, text, rng):
        tier = "strong-frontier" if "hard" in text.lower() else "cheap-small"
        return RouteDecision(selected_candidate=tier, confidence=1.0, fallback_used=False)


def _fake_completion(model=None, messages=None, **kw):
    class R:
        def model_dump(self_inner):
            return {"id": "c", "model": model,
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    return R()


class TestOfflineE2E(unittest.TestCase):
    def test_mixed_tier_trajectory_and_rollup(self):
        trace = "/tmp/tau2_e2e_trace.jsonl"
        open(trace, "w").close()
        app = build_proxy_app({"hardword": _HardWordRouter()}, trace,
                              litellm_completion=_fake_completion)
        client = TestClient(app)
        client.post("/begin_task", json={"router": "hardword", "benchmark": "tau2-bench (live)",
                                         "task_id": "tau2-1", "trial": 0})
        client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "an easy request"}]})
        client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "now a hard multi-step ask"}]})
        with open(trace) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["chosen_tier"] for r in rows], ["cheap-small", "strong-frontier"])
        self.assertEqual(_escalation_count(rows), 1)
        cost, mix = _rollup_cost_from_steps(trace, "tau2-1", 0)
        self.assertGreater(cost, 0)
        self.assertEqual(mix, {"cheap-small": 1, "strong-frontier": 1})


if __name__ == "__main__":
    unittest.main()
