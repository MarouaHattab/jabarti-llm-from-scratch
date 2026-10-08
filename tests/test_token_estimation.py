import unittest
from unittest.mock import patch

import pandas as pd


class TokenEstimationTests(unittest.TestCase):
    def test_finetune_sums_lengths_before_truncating_estimate(self):
        from data_exploration.token_estimation import analyze, estimate_tokens
        frame = pd.DataFrame({"question": ["abc", "abcde"], "answer": ["1234", "123456"]})
        expected = ((frame["question"].str.len() + frame["answer"].str.len()) / 3.5).astype(int)
        pd.testing.assert_series_equal(estimate_tokens(frame, "finetune"), expected, check_names=False)
        self.assertIn(expected.describe().round(1).to_string(), analyze(frame, "finetune"))

    def test_context_boundary_is_inclusive(self):
        from data_exploration.token_estimation import analyze
        frame = pd.DataFrame({"text": ["x" * 1795, "x" * 1796]})
        report = analyze(frame, "pretrain")
        self.assertIn("max_length=512 | kept: 50.0% | dropped: 50.0%", report)
        self.assertIn("estimates", report)

    def test_invalid_incomplete_and_empty_samples_are_explained(self):
        from data_exploration.token_estimation import analyze, estimate_tokens
        frame = pd.DataFrame({"question": ["abc", None, 5], "answer": ["1234", "x", "y"]})
        self.assertEqual(estimate_tokens(frame, "finetune").tolist(), [2])
        self.assertIn("Samples excluded: 2", analyze(frame, "finetune"))
        self.assertIn("No valid samples", analyze(pd.DataFrame(), "pretrain"))

    def test_invalid_constant_fails_clearly(self):
        from data_exploration.token_estimation import estimate_tokens
        with patch("data_exploration.config.CHARS_PER_TOKEN", 0):
            with self.assertRaisesRegex(ValueError, "CHARS_PER_TOKEN"):
                estimate_tokens(pd.DataFrame({"text": ["abc"]}), "pretrain")



if __name__ == "__main__":
    unittest.main()
