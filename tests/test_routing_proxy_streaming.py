import json
import unittest
from fastapi.testclient import TestClient
from router_benchmark.interfaces import RouteDecision
from router_benchmark.live.routing_context import PerRequestRouter
from router_benchmark.live.routing_proxy import build_proxy_app


class _StubRouter(PerRequestRouter):
    name = "stub"
    def _route_on_text(self, text, rng):
        return RouteDecision(selected_candidate="mid-general", confidence=0.5, fallback_used=False)


class _Chunk:
    def __init__(self, d): self._d = d
    def model_dump(self): return self._d


def _fake_stream_completion(with_usage=True):
    def _fn(model=None, messages=None, stream=False, stream_options=None, **kw):
        assert stream is True
        assert stream_options == {"include_usage": True}
        chunks = [
            _Chunk({"choices": [{"delta": {"content": "Hel"}}]}),
            _Chunk({"choices": [{"delta": {"content": "lo"}}]}),
        ]
        if with_usage:
            chunks.append(_Chunk({"choices": [{"delta": {}}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
        return iter(chunks)
    return _fn


class TestRoutingProxyStreaming(unittest.TestCase):
    def _run(self, with_usage):
        trace = "/tmp/router_proxy_stream_trace.jsonl"
        open(trace, "w").close()
        app = build_proxy_app({"stub": _StubRouter()}, trace,
                              litellm_completion=_fake_stream_completion(with_usage))
        client = TestClient(app)
        client.post("/begin_task", json={"router": "stub", "benchmark": "tau2",
                                         "task_id": "t1", "trial": 0})
        with client.stream("POST", "/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
            text = "".join(part for part in r.iter_text())
        rows = [json.loads(l) for l in open(trace) if l.strip()]
        return text, rows[0]

    def test_sse_passthrough_and_usage(self):
        text, row = self._run(with_usage=True)
        self.assertIn('data: {"choices"', text)
        self.assertIn("data: [DONE]", text)
        self.assertEqual(row["input_tokens"], 7)
        self.assertEqual(row["output_tokens"], 3)
        self.assertFalse(row["usage_estimated"])

    def test_missing_usage_flagged_estimated_not_zero_cost_silent(self):
        text, row = self._run(with_usage=False)
        self.assertIn("data: [DONE]", text)
        self.assertTrue(row["usage_estimated"])
        self.assertEqual(row["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
