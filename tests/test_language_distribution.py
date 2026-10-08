import unittest

import pandas as pd


class LanguageTests(unittest.TestCase):
    def test_percentages_use_total_rows_including_null_languages(self):
        from data_exploration.language_distribution import analyze
        report = analyze(pd.DataFrame({"language": ["en", "ar", "en", None]}), "pretrain")
        self.assertIn("en => 2 samples -- 50.0%", report)
        self.assertIn("ar => 1 samples -- 25.0%", report)
        self.assertIn("Missing language labels: 1", report)

    def test_missing_language_column_and_empty_split(self):
        from data_exploration.language_distribution import analyze
        self.assertIn("Skipped", analyze(pd.DataFrame({"text": ["abc"]}), "pretrain"))
        self.assertIn("No rows", analyze(pd.DataFrame({"language": []}), "pretrain"))



if __name__ == "__main__":
    unittest.main()
