import unittest
from router_benchmark.interfaces import RouteDecision
from router_benchmark.live.routing_context import latest_routing_text, PerRequestRouter


class TestLatestRoutingText(unittest.TestCase):
    def test_returns_last_user_message(self):
        msgs = [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "route on me"},
        ]
        self.assertEqual(latest_routing_text(msgs), "route on me")

    def test_prefers_tool_result_over_older_user(self):
        msgs = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "tool output here"},
        ]
        self.assertEqual(latest_routing_text(msgs), "tool output here")

    def test_handles_openai_content_parts(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "parted"}]}]
        self.assertEqual(latest_routing_text(msgs), "parted")

    def test_empty_messages_returns_empty_string(self):
        self.assertEqual(latest_routing_text([]), "")


class _FakeRouter(PerRequestRouter):
    def _route_on_text(self, text, rng):
        return RouteDecision(selected_candidate="cheap-small", confidence=len(text) / 100, fallback_used=False)


class TestPerRequestRouterMixin(unittest.TestCase):
    def test_route_request_delegates_to_route_on_text(self):
        r = _FakeRouter()
        d = r.route_request(messages=[{"role": "user", "content": "hello"}], tools=None,
                            candidates=None, context={}, rng=0)
        self.assertEqual(d.selected_candidate, "cheap-small")
        self.assertAlmostEqual(d.confidence, 0.05)


if __name__ == "__main__":
    unittest.main()
