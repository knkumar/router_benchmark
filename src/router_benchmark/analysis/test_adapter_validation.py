"""Adapter-validation tests for the two adapters that had bugs
(paper1.tex Section sec:adapter-bugs / Appendix app:adapter-bugs).

These prove each adapter's route() calls the package's real decision
path rather than a harness-side proxy -- the exact property whose
absence caused both original bugs. No live API calls are made: the
underlying network/embedding calls are mocked so this runs offline and
at zero cost, per Phase A's no-new-experiment-cost constraint.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestLiteLLMRouterLiveCallsRealPath(unittest.TestCase):
    """LiteLLM Router bug: adapter built a real litellm.Router but never
    called it, returning a hardcoded difficulty-threshold ladder instead.
    This test fails against that old code (route() would never touch
    self._router) and passes against the fixed adapter."""

    @staticmethod
    def _async_return(value):
        async def _coro(*args, **kwargs):
            return value
        return _coro

    def test_route_calls_async_get_available_deployment(self):
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.live_routers import LiteLLMRouterLive

        adapter = LiteLLMRouterLive.__new__(LiteLLMRouterLive)
        mock_router = MagicMock()
        mock_router.async_get_available_deployment = MagicMock(
            side_effect=self._async_return({"model_info": {"id": "cheap-small"}})
        )
        adapter._router = mock_router

        task = Task(
            task_id="t0",
            benchmark_name="unit-test",
            domain=TaskDomain.QA_REASONING,
            difficulty=0.5,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "What is 2+2?"},
        )
        decision = adapter.route(task, context={}, rng=None)

        mock_router.async_get_available_deployment.assert_called_once()
        _, kwargs = mock_router.async_get_available_deployment.call_args
        self.assertEqual(kwargs["model"], adapter._MODEL_GROUP)
        self.assertIn("What is 2+2?", kwargs["messages"][0]["content"])
        self.assertEqual(decision.selected_candidate, "cheap-small")

    def test_route_reflects_mocked_deployment_not_a_fixed_ladder(self):
        """If a hardcoded ladder were still in place, changing the mocked
        deployment id would have no effect on the returned decision."""
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.live_routers import LiteLLMRouterLive

        adapter = LiteLLMRouterLive.__new__(LiteLLMRouterLive)
        mock_router = MagicMock()
        mock_router.async_get_available_deployment = MagicMock(
            side_effect=self._async_return({"model_info": {"id": "strong-frontier"}})
        )
        adapter._router = mock_router

        task = Task(
            task_id="t1",
            benchmark_name="unit-test",
            domain=TaskDomain.QA_REASONING,
            difficulty=0.9,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "trivial arithmetic that a difficulty ladder would route cheap"},
        )
        decision = adapter.route(task, context={}, rng=None)
        self.assertEqual(decision.selected_candidate, "strong-frontier")


class TestAurelioSemanticRouterLiveCallsRealPath(unittest.TestCase):
    """Aurelio bug: adapter fabricated confidence=0.7 on a sub-threshold
    match. This test verifies (a) the router is constructed with
    aggregation="max" (the real fix, not "mean"), and (b) route() reports
    the package's own real fallback_used/confidence when no route
    clears threshold, rather than substituting a fabricated match."""

    def test_router_constructed_with_max_aggregation(self):
        from router_benchmark.live.live_routers import AurelioSemanticRouterLive

        fake_semantic_router_module = types.ModuleType("semantic_router")
        fake_encoders_module = types.ModuleType("semantic_router.encoders")
        fake_routers_module = types.ModuleType("semantic_router.routers")

        captured_kwargs = {}

        class FakeRoute:
            def __init__(self, name, utterances):
                self.name = name
                self.utterances = utterances

        class FakeOpenAIEncoder:
            def __init__(self, name):
                self.name = name

        class FakeSemanticRouter:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        fake_semantic_router_module.Route = FakeRoute
        fake_encoders_module.OpenAIEncoder = FakeOpenAIEncoder
        fake_routers_module.SemanticRouter = FakeSemanticRouter

        with patch.dict(sys.modules, {
            "semantic_router": fake_semantic_router_module,
            "semantic_router.encoders": fake_encoders_module,
            "semantic_router.routers": fake_routers_module,
        }):
            AurelioSemanticRouterLive()

        self.assertEqual(captured_kwargs.get("aggregation"), "max")

    def test_route_reports_honest_fallback_on_no_match(self):
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.live_routers import AurelioSemanticRouterLive

        adapter = AurelioSemanticRouterLive.__new__(AurelioSemanticRouterLive)
        mock_router = MagicMock(return_value=None)  # no route cleared threshold
        adapter._router = mock_router

        task = Task(
            task_id="t2",
            benchmark_name="unit-test",
            domain=TaskDomain.WEB_NAVIGATION,
            difficulty=0.5,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "navigate to the checkout page and click submit"},
        )
        decision = adapter.route(task, context={}, rng=None)

        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.confidence, 0.0)
        self.assertNotEqual(decision.confidence, 0.7)  # the old fabricated value


class TestRouteLLMLiveCallsRealPath(unittest.TestCase):
    """RouteLLM adapter: route() must call the real SWRankingRouter's
    calculate_strong_win_rate() and use RouteLLM's own binary threshold
    rule (Controller.route: strong iff win rate >= threshold), not a
    hardcoded tier. Verified by changing the mocked win rate and
    confirming the returned tier and confidence change accordingly."""

    def test_route_calls_calculate_strong_win_rate(self):
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.routellm_live import RouteLLMLive

        adapter = RouteLLMLive.__new__(RouteLLMLive)
        adapter.threshold = 0.5
        adapter._router = MagicMock()
        adapter._router.calculate_strong_win_rate = MagicMock(return_value=0.9)

        task = Task(
            task_id="t0",
            benchmark_name="unit-test",
            domain=TaskDomain.QA_REASONING,
            difficulty=0.9,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "a genuinely hard reasoning question"},
        )
        decision = adapter.route(task, context={}, rng=None)

        adapter._router.calculate_strong_win_rate.assert_called_once_with(task.metadata["prompt"])
        self.assertEqual(decision.selected_candidate, "strong-frontier")
        self.assertAlmostEqual(decision.confidence, 0.9)

    def test_route_flips_tier_when_mocked_winrate_crosses_threshold(self):
        """If a hardcoded tier were in place, changing the mocked win rate
        would have no effect on the returned decision."""
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.routellm_live import RouteLLMLive

        adapter = RouteLLMLive.__new__(RouteLLMLive)
        adapter.threshold = 0.5
        adapter._router = MagicMock()
        adapter._router.calculate_strong_win_rate = MagicMock(return_value=0.1)

        task = Task(
            task_id="t1",
            benchmark_name="unit-test",
            domain=TaskDomain.QA_REASONING,
            difficulty=0.1,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "trivial arithmetic"},
        )
        decision = adapter.route(task, context={}, rng=None)

        self.assertEqual(decision.selected_candidate, "cheap-small")
        self.assertAlmostEqual(decision.confidence, 0.9)  # 1 - 0.1


class TestVLLMSemanticRouterLiveCallsRealPath(unittest.TestCase):
    """vLLM Semantic Router adapter: route() must call the real Envoy
    endpoint and read the real x-vsr-selected-model / x-vsr-response-path
    response headers, not a hardcoded tier. Verified by changing the
    mocked response headers and confirming the returned decision changes."""

    def test_route_calls_envoy_and_reads_selected_model_header(self):
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive

        adapter = VLLMSemanticRouterLive.__new__(VLLMSemanticRouterLive)
        adapter.envoy_url = "http://localhost:8909/v1/chat/completions"
        adapter.timeout_s = 30.0

        mock_response = MagicMock()
        mock_response.headers = {"x-vsr-selected-model": "cheap-small", "x-vsr-response-path": "direct"}
        mock_response.raise_for_status = MagicMock()

        task = Task(
            task_id="t0",
            benchmark_name="unit-test",
            domain=TaskDomain.QA_REASONING,
            difficulty=0.2,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "What is the capital of France?"},
        )

        with patch("router_benchmark.live.vllm_sr_live.requests.post", return_value=mock_response) as mock_post:
            decision = adapter.route(task, context={}, rng=None)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["messages"][0]["content"], task.metadata["prompt"])
        self.assertEqual(decision.selected_candidate, "cheap-small")
        self.assertFalse(decision.fallback_used)

    def test_route_reflects_mocked_header_change_not_a_fixed_tier(self):
        """If a hardcoded tier were in place, changing the mocked header
        would have no effect on the returned decision."""
        from router_benchmark.interfaces import Task, TaskDomain
        from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive

        adapter = VLLMSemanticRouterLive.__new__(VLLMSemanticRouterLive)
        adapter.envoy_url = "http://localhost:8909/v1/chat/completions"
        adapter.timeout_s = 30.0

        mock_response = MagicMock()
        mock_response.headers = {"x-vsr-selected-model": "strong-frontier", "x-vsr-response-path": "fallback"}
        mock_response.raise_for_status = MagicMock()

        task = Task(
            task_id="t1",
            benchmark_name="unit-test",
            domain=TaskDomain.QA_REASONING,
            difficulty=0.9,
            requires_tool_call=False,
            candidates=(),
            metadata={"prompt": "a much harder question"},
        )

        with patch("router_benchmark.live.vllm_sr_live.requests.post", return_value=mock_response):
            decision = adapter.route(task, context={}, rng=None)

        self.assertEqual(decision.selected_candidate, "strong-frontier")
        self.assertTrue(decision.fallback_used)


if __name__ == "__main__":
    unittest.main()

