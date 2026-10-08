import unittest

import pandas as pd


class MissingValuesTests(unittest.TestCase):
    def test_null_counts_and_percentages(self):
        from data_exploration.missing_values import analyze
        report = analyze(pd.DataFrame({"text": [None, "abc"], "language": [None, None]}), "pretrain")
        self.assertIn("text: 1 (50.0%)", report)
        self.assertIn("language: 2 (100.0%)", report)

    def test_no_missing_and_empty_data(self):
        from data_exploration.missing_values import analyze
        self.assertIn("NONE", analyze(pd.DataFrame({"text": ["abc"]}), "pretrain"))
        self.assertIn("N/A", analyze(pd.DataFrame({"text": []}), "pretrain"))



if __name__ == "__main__":
    unittest.main()
