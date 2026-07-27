import json
import unittest
from fastapi.testclient import TestClient
from router_benchmark.interfaces import RouteDecision
from router_benchmark.live.routing_context import PerRequestRouter
from router_benchmark.live.routing_proxy import build_proxy_app


class _StubRouter(PerRequestRouter):
    name = "stub"
    def __init__(self, tier): self._tier = tier
    def _route_on_text(self, text, rng):
        return RouteDecision(selected_candidate=self._tier, confidence=0.9, fallback_used=False)


def _fake_completion(model=None, messages=None, **kw):
    # Mimic a litellm ModelResponse's .model_dump()
    class R:
        def model_dump(self_inner):
            return {"id": "cmpl-1", "model": model, "choices":
                    [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    _fake_completion.last_kwargs = {"model": model, **kw}
    return R()


class TestRoutingProxyNonStreaming(unittest.TestCase):
    def setUp(self):
        self.trace = "/tmp/router_proxy_test_trace.jsonl"
        open(self.trace, "w").close()
        app = build_proxy_app({"stub": _StubRouter("strong-frontier")}, self.trace,
                              litellm_completion=_fake_completion)
        self.client = TestClient(app)

    def test_routes_forwards_and_logs_step(self):
        self.client.post("/begin_task", json={"router": "stub", "benchmark": "tau2",
                                              "task_id": "t1", "trial": 0})
        resp = self.client.post("/v1/chat/completions",
                                json={"messages": [{"role": "user", "content": "solve this"}],
                                      "temperature": 1.0, "max_tokens": 64})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "hi")
        # forwarded to the strong tier's real backend model
        self.assertEqual(_fake_completion.last_kwargs["model"], "anthropic/claude-opus-4-8")
        rows = [json.loads(l) for l in open(self.trace) if l.strip()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["chosen_tier"], "strong-frontier")
        self.assertEqual(row["task_id"], "t1")
        self.assertEqual(row["step_idx"], 0)
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["output_tokens"], 5)
        # cost = 10*5/1e6 + 5*25/1e6
        self.assertAlmostEqual(row["cost_usd"], 10 * (5.0 / 1e6) + 5 * (25.0 / 1e6))
        self.assertFalse(row["usage_estimated"])

    def test_step_idx_increments_and_resets_on_begin_task(self):
        self.client.post("/begin_task", json={"router": "stub", "benchmark": "tau2",
                                              "task_id": "t1", "trial": 0})
        self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "a"}]})
        self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "b"}]})
        self.client.post("/begin_task", json={"router": "stub", "benchmark": "tau2",
                                              "task_id": "t2", "trial": 0})
        self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "c"}]})
        rows = [json.loads(l) for l in open(self.trace) if l.strip()]
        self.assertEqual([r["step_idx"] for r in rows], [0, 1, 0])
        self.assertEqual([r["task_id"] for r in rows], ["t1", "t1", "t2"])

    def test_forced_tier_overrides_router(self):
        app = build_proxy_app({"stub": _StubRouter("strong-frontier")}, self.trace,
                              forced_tier="cheap-small", litellm_completion=_fake_completion)
        client = TestClient(app)
        client.post("/begin_task", json={"router": "stub", "benchmark": "tau2",
                                         "task_id": "t1", "trial": 0})
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(_fake_completion.last_kwargs["model"], "openai/gpt-5.4-nano")

    def test_missing_usage_sets_estimated_true(self):
        def _fake_completion_no_usage(model=None, messages=None, **kw):
            class R:
                def model_dump(self_inner):
                    return {"id": "cmpl-1", "model": model, "choices":
                            [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
                    # No "usage" key
            return R()
        
        trace = "/tmp/router_proxy_test_trace_no_usage.jsonl"
        open(trace, "w").close()
        app = build_proxy_app({"stub": _StubRouter("strong-frontier")}, trace,
                              litellm_completion=_fake_completion_no_usage)
        client = TestClient(app)
        client.post("/begin_task", json={"router": "stub", "benchmark": "tau2",
                                         "task_id": "t1", "trial": 0})
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "test"}]})
        self.assertEqual(resp.status_code, 200)
        rows = [json.loads(l) for l in open(trace) if l.strip()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["input_tokens"], 0)
        self.assertEqual(row["output_tokens"], 0)
        self.assertTrue(row["usage_estimated"])

    def test_unknown_router_returns_400_without_forced_tier(self):
        """Posting /begin_task with unknown router name should return 400 error."""
        resp = self.client.post("/begin_task", json={"router": "unknown-router", 
                                                      "benchmark": "tau2",
                                                      "task_id": "t1", "trial": 0})
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("error", body)
        self.assertIn("unknown-router", body["error"])
        self.assertIn("valid", body["error"])
        self.assertIn("stub", body["error"])

    def test_unknown_router_allowed_with_forced_tier(self):
        """Posting /begin_task with unknown router name should NOT error when forced_tier is set."""
        app = build_proxy_app({"stub": _StubRouter("strong-frontier")}, self.trace,
                              forced_tier="cheap-small", litellm_completion=_fake_completion)
        client = TestClient(app)
        # Should succeed even with unknown router name because forced_tier mode never uses active_router
        resp = client.post("/begin_task", json={"router": "unknown-router",
                                                 "benchmark": "tau2",
                                                 "task_id": "t1", "trial": 0})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})


if __name__ == "__main__":
    unittest.main()
