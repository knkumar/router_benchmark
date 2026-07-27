import json
import os
import unittest
from router_benchmark.live.tau2_live import _rollup_cost_from_steps, _tier_mix, _escalation_count


class TestStepRollup(unittest.TestCase):
    def setUp(self):
        self.trace = "/tmp/tau2_steps_test.jsonl"
        rows = [
            {"task_id": "tau2-1", "trial": 0, "step_idx": 0, "chosen_tier": "cheap-small", "cost_usd": 0.001},
            {"task_id": "tau2-1", "trial": 0, "step_idx": 1, "chosen_tier": "strong-frontier", "cost_usd": 0.02},
            {"task_id": "tau2-1", "trial": 0, "step_idx": 2, "chosen_tier": "cheap-small", "cost_usd": 0.001},
            {"task_id": "tau2-1", "trial": 1, "step_idx": 0, "chosen_tier": "mid-general", "cost_usd": 0.005},
            {"task_id": "tau2-2", "trial": 0, "step_idx": 0, "chosen_tier": "mid-general", "cost_usd": 0.005},
        ]
        with open(self.trace, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_rollup_sums_only_matching_task_trial(self):
        cost, tier_mix = _rollup_cost_from_steps(self.trace, "tau2-1", 0)
        self.assertAlmostEqual(cost, 0.022)
        self.assertEqual(tier_mix, {"cheap-small": 2, "strong-frontier": 1})

    def test_rollup_filters_by_trial(self):
        # Verify that trial filtering actually works: a row with the same task_id
        # but different trial should not be included in the rollup.
        cost, tier_mix = _rollup_cost_from_steps(self.trace, "tau2-1", 1)
        self.assertAlmostEqual(cost, 0.005)
        self.assertEqual(tier_mix, {"mid-general": 1})

    def test_tier_mix_and_escalation(self):
        # Explicitly filter to trial=0 to test a specific trial's rows
        with open(self.trace) as f:
            rows = [json.loads(l) for l in f if json.loads(l).get("task_id") == "tau2-1" and json.loads(l).get("trial", 0) == 0]
        self.assertEqual(_escalation_count(rows), 1)  # cheap -> strong once
        self.assertEqual(_tier_mix(rows), {"cheap-small": 2, "strong-frontier": 1})

    def test_missing_trace_file_returns_zero_cost(self):
        """Calling _rollup_cost_from_steps with non-existent file should return (0.0, {})."""
        missing_trace = "/tmp/nonexistent_trace_file_12345.jsonl"
        # Ensure the file doesn't exist
        if os.path.exists(missing_trace):
            os.remove(missing_trace)
        
        cost, tier_mix = _rollup_cost_from_steps(missing_trace, "tau2-1", 0)
        self.assertEqual(cost, 0.0)
        self.assertEqual(tier_mix, {})


if __name__ == "__main__":
    unittest.main()
