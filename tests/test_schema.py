import unittest

import pandas as pd


class SchemaTests(unittest.TestCase):
    def test_reports_all_columns_and_types(self):
        from data_exploration.schema import analyze
        frame = pd.DataFrame({"text": ["abc"], "chunk_index": [1]})
        report = analyze(frame, "pretrain")
        self.assertIn("Columns: 2", report)
        for column, dtype in frame.dtypes.items():
            self.assertIn(f"{column}: {dtype}", report)

    def test_empty_schema_is_explicit(self):
        from data_exploration.schema import analyze
        self.assertIn("No columns", analyze(pd.DataFrame(), "pretrain"))



if __name__ == "__main__":
    unittest.main()
