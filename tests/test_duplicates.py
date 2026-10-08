import unittest

import pandas as pd


class DuplicateTests(unittest.TestCase):
    def test_duplicate_counts_exclude_first_occurrence(self):
        from data_exploration.duplicates import analyze
        frame = pd.DataFrame({"text": ["abc", "abc", "def"],
                              "article_id": ["1", "1", "2"], "chunk_index": [0, 0, 0]})
        report = analyze(frame, "pretrain")
        self.assertIn("Duplicate text records: 1 (33.3%)", report)
        self.assertIn("Duplicate (article_id, chunk_index) pairs: 1 (33.3%)", report)

    def test_unsupported_schema_is_skipped(self):
        from data_exploration.duplicates import analyze
        report = analyze(pd.DataFrame({"question": ["abc"], "answer": ["def"]}), "finetune")
        self.assertIn("Skipped", report)
        self.assertNotIn("Duplicate text records: 0", report)

    def test_empty_dataset_has_no_percentage(self):
        from data_exploration.duplicates import analyze
        self.assertIn("N/A", analyze(pd.DataFrame({"text": []}), "pretrain"))



if __name__ == "__main__":
    unittest.main()
