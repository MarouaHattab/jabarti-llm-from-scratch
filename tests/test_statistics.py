import unittest

import pandas as pd


class StatisticsTests(unittest.TestCase):
    def test_counts_samples_without_mutating_frame(self):
        from data_exploration.statistics import analyze
        frame = pd.DataFrame({"text": ["abc", "def", "ghi"]})
        original = frame.copy(deep=True)
        self.assertIn("Samples: 3", analyze(frame, "pretrain"))
        pd.testing.assert_frame_equal(frame, original)
        self.assertIn("Samples: 0", analyze(frame.iloc[:0], "pretrain"))



if __name__ == "__main__":
    unittest.main()
