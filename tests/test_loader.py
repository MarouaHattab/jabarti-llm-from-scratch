import unittest
from unittest.mock import patch

import pandas as pd

from data_exploration import config
from data_exploration.loader import load_split


class LoaderTests(unittest.TestCase):
    def test_loads_only_requested_split_without_version_options(self):
        expected = pd.DataFrame({"text": ["hello"]})
        with patch("data_exploration.loader.load_dataset") as remote:
            remote.return_value.to_pandas.return_value = expected
            actual = load_split("pretrain", "phase1_train")
            self.assertIs(actual, expected)
            args, kwargs = remote.call_args
            self.assertEqual(args, (config.REPO, "pretrain"))
            self.assertNotIn("revision", kwargs)
            self.assertEqual(kwargs["split"], "phase1_train")
            self.assertEqual(kwargs["data_files"], {
                "phase1_train": "pretrain/phase1_train-*.parquet",
            })
            self.assertNotIn("streaming", kwargs)

    def test_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset split"):
            load_split("pretrain", "unknown")

    def test_error_includes_split_and_original_reason(self):
        with patch("data_exploration.loader.load_dataset", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "pretrain / phase1_train.*offline"):
                load_split("pretrain", "phase1_train")

    def test_empty_dataset_preserves_schema(self):
        expected = pd.DataFrame(columns=["text", "language"])
        with patch("data_exploration.loader.load_dataset") as remote:
            remote.return_value.to_pandas.return_value = expected
            frame = load_split("pretrain", "phase1_train")
            self.assertTrue(frame.empty)
            self.assertEqual(list(frame), ["text", "language"])


if __name__ == "__main__":
    unittest.main()
