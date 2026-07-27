# router_benchmark/tests/test_per_request_router_adapters.py
"""route_request must route on the latest turn via the SAME package path as
route(task). Package calls are mocked so this runs offline (mirrors
analysis/test_adapter_validation.py)."""
import unittest
from unittest.mock import MagicMock, patch
from router_benchmark.interfaces import Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES


def _task(prompt):
    return Task(task_id="x", benchmark_name="b", domain=TaskDomain.MULTI_TURN_POLICY,
                difficulty=0.5, requires_tool_call=True, candidates=LIVE_CANDIDATES,
                metadata={"prompt": prompt})


class TestRouteLLMPerRequest(unittest.TestCase):
    def test_route_request_uses_latest_turn_via_real_winrate_call(self):
        from router_benchmark.live.routellm_live import RouteLLMLive
        r = RouteLLMLive.__new__(RouteLLMLive)          # bypass heavy __init__
        r.threshold = 0.5
        r._router = MagicMock()
        r._router.calculate_strong_win_rate.return_value = 0.9
        msgs = [{"role": "user", "content": "easy"}, {"role": "assistant", "content": "..."},
                {"role": "user", "content": "very hard multi-step proof"}]
        d = r.route_request(messages=msgs, tools=None, candidates=LIVE_CANDIDATES, context={}, rng=0)
        r._router.calculate_strong_win_rate.assert_called_once_with("very hard multi-step proof")
        self.assertEqual(d.selected_candidate, "strong-frontier")

    def test_route_task_still_works(self):
        from router_benchmark.live.routellm_live import RouteLLMLive
        r = RouteLLMLive.__new__(RouteLLMLive)
        r.threshold = 0.5
        r._router = MagicMock()
        r._router.calculate_strong_win_rate.return_value = 0.1
        d = r.route(_task("simple"), context={}, rng=0)
        self.assertEqual(d.selected_candidate, "cheap-small")


class TestAurelioPerRequest(unittest.TestCase):
    def test_route_request_calls_semantic_router_on_latest_turn(self):
        from router_benchmark.live.live_routers import AurelioSemanticRouterLive
        r = AurelioSemanticRouterLive.__new__(AurelioSemanticRouterLive)
        r._router = MagicMock()
        r._router.return_value = MagicMock(name="mid-general", similarity_score=0.8)
        r._router.return_value.name = "mid-general"
        msgs = [{"role": "user", "content": "explain a moderately complex concept"}]
        d = r.route_request(messages=msgs, tools=None, candidates=LIVE_CANDIDATES, context={}, rng=0)
        r._router.assert_called_once_with("explain a moderately complex concept")
        self.assertEqual(d.selected_candidate, "mid-general")


class TestLiteLLMPerRequest(unittest.TestCase):
    def test_route_request_content_blind_returns_cheapest(self):
        from router_benchmark.live.live_routers import LiteLLMRouterLive
        r = LiteLLMRouterLive.__new__(LiteLLMRouterLive)
        r._MODEL_GROUP = "agentic-tiers"
        async def _dep(**kw):
            return {"model_info": {"id": "cheap-small"}}
        r._router = MagicMock()
        r._router.async_get_available_deployment = MagicMock(side_effect=_dep)
        msgs = [{"role": "user", "content": "easy"}, {"role": "assistant", "content": "..."},
                {"role": "user", "content": "latest turn content"}]
        d = r.route_request(messages=msgs,
                            tools=None, candidates=LIVE_CANDIDATES, context={}, rng=0)
        r._router.async_get_available_deployment.assert_called_once()
        _, kwargs = r._router.async_get_available_deployment.call_args
        self.assertEqual(kwargs["model"], "agentic-tiers")
        self.assertEqual(kwargs["messages"][0]["content"], "latest turn content")
        self.assertEqual(d.selected_candidate, "cheap-small")


class TestVLLMSemanticRouterPerRequest(unittest.TestCase):
    def test_route_request_calls_envoy_on_latest_turn(self):
        from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
        r = VLLMSemanticRouterLive.__new__(VLLMSemanticRouterLive)
        r.envoy_url = "http://localhost:8909/v1/chat/completions"
        r.timeout_s = 30.0
        
        mock_response = MagicMock()
        mock_response.headers = {"x-vsr-selected-model": "strong-frontier", "x-vsr-response-path": "direct"}
        mock_response.raise_for_status = MagicMock()
        
        msgs = [{"role": "user", "content": "what is easy"}, {"role": "assistant", "content": "..."},
                {"role": "user", "content": "now a harder question"}]
        
        with patch("router_benchmark.live.vllm_sr_live.requests.post", return_value=mock_response) as mock_post:
            d = r.route_request(messages=msgs, tools=None, candidates=LIVE_CANDIDATES, context={}, rng=0)
        
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["messages"][0]["content"], "now a harder question")
        self.assertEqual(d.selected_candidate, "strong-frontier")
        self.assertFalse(d.fallback_used)


if __name__ == "__main__":
    unittest.main()
