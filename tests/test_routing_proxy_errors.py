import json
import unittest
from fastapi.testclient import TestClient
from router_benchmark.interfaces import RouteDecision
from router_benchmark.live.routing_context import PerRequestRouter
from router_benchmark.live.routing_proxy import build_proxy_app


class _StubRouter(PerRequestRouter):
    name = "stub"
    def _route_on_text(self, text, rng):
        return RouteDecision(selected_candidate="strong-frontier", confidence=1.0, fallback_used=False)


class _Once:
    """Fails the first call with a 400-shaped error, succeeds the second."""
    def __init__(self): self.calls = 0
    def __call__(self, model=None, messages=None, **kw):
        self.calls += 1
        if self.calls == 1:
            class BadRequestError(Exception):
                status_code = 400
            raise BadRequestError("temperature deprecated")
        class R:
            def model_dump(self_inner):
                return {"id": "c", "model": model, "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return R()


class _AlwaysBad:
    def __call__(self, model=None, messages=None, **kw):
        class BadRequestError(Exception):
            status_code = 400
        raise BadRequestError("still bad")


class TestProxyErrorHandling(unittest.TestCase):
    def test_retries_once_same_tier_then_succeeds(self):
        trace = "/tmp/proxy_err_trace.jsonl"; open(trace, "w").close()
        fake = _Once()
        app = build_proxy_app({"stub": _StubRouter()}, trace, litellm_completion=fake)
        c = TestClient(app)
        c.post("/begin_task", json={"router": "stub", "benchmark": "b", "task_id": "t", "trial": 0})
        r = c.post("/v1/chat/completions",
                   json={"messages": [{"role": "user", "content": "x"}], "temperature": 0.0})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(fake.calls, 2)  # retried once
        row = [json.loads(l) for l in open(trace) if l.strip()][0]
        self.assertEqual(row["chosen_tier"], "strong-frontier")  # tier unchanged

    def test_persistent_400_surfaces_502_and_logs_error(self):
        trace = "/tmp/proxy_err_trace2.jsonl"; open(trace, "w").close()
        app = build_proxy_app({"stub": _StubRouter()}, trace, litellm_completion=_AlwaysBad())
        c = TestClient(app)
        c.post("/begin_task", json={"router": "stub", "benchmark": "b", "task_id": "t", "trial": 0})
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(r.status_code, 502)
        row = [json.loads(l) for l in open(trace) if l.strip()][0]
        self.assertIn("error", row)
        self.assertEqual(row["chosen_tier"], "strong-frontier")


if __name__ == "__main__":
    unittest.main()
