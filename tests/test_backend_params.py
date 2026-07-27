import unittest
from router_benchmark.live.backend_params import sanitize_params


class TestSanitizeParams(unittest.TestCase):
    def test_drops_zero_temperature_for_opus(self):
        out = sanitize_params("anthropic/claude-opus-4-8", {"temperature": 0.0, "max_tokens": 100})
        self.assertNotIn("temperature", out)
        self.assertEqual(out["max_tokens"], 100)

    def test_keeps_nonzero_temperature_for_opus(self):
        out = sanitize_params("anthropic/claude-opus-4-8", {"temperature": 1.0})
        self.assertEqual(out["temperature"], 1.0)

    def test_renames_max_tokens_for_gpt5(self):
        out = sanitize_params("openai/gpt-5.4-nano", {"max_tokens": 256})
        self.assertNotIn("max_tokens", out)
        self.assertEqual(out["max_completion_tokens"], 256)

    def test_sonnet_untouched(self):
        params = {"temperature": 0.0, "max_tokens": 100}
        out = sanitize_params("anthropic/claude-sonnet-4-6", params)
        self.assertEqual(out, params)

    def test_does_not_mutate_input(self):
        params = {"temperature": 0.0}
        sanitize_params("anthropic/claude-opus-4-8", params)
        self.assertEqual(params, {"temperature": 0.0})


if __name__ == "__main__":
    unittest.main()
