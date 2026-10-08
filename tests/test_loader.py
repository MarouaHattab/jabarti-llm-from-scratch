import unittest
from unittest.mock import patch

import pandas as pd

from data_exploration import config
from data_exploration.loader import load_split


class LoaderTests(unittest.TestCase):
    def test_loads_only_requested_split_at_default_revision(self):
        expected = pd.DataFrame({"text": ["hello"]})
        with patch("data_exploration.loader.load_dataset") as remote:
            remote.return_value.to_pandas.return_value = expected
            actual = load_split("pretrain", "phase1_train")
            self.assertIs(actual, expected)
            args, kwargs = remote.call_args
            self.assertEqual(args, (config.REPO, "pretrain"))
            self.assertEqual(kwargs["revision"], "main")
            self.assertEqual(kwargs["split"], "phase1_train")
            self.assertEqual(kwargs["data_files"], {
                "phase1_train": "pretrain/phase1_train-*.parquet",
            })
            self.assertFalse(kwargs["streaming"])

    def test_accepts_optional_revision_setting(self):
        with patch.object(config, "REVISION", "specific-revision", create=True), patch("data_exploration.loader.load_dataset") as remote:
            load_split("pretrain", "phase1_train")
            self.assertEqual(remote.call_args.kwargs["revision"], "specific-revision")

    def test_preview_streams_a_deterministic_prefix(self):
        with patch("data_exploration.loader.load_dataset") as remote:
            remote.return_value.take.return_value = [{"text": "first"}, {"text": "second"}]
            frame = load_split("pretrain", "phase1_train", max_rows=2)
            self.assertTrue(remote.call_args.kwargs["streaming"])
            remote.return_value.take.assert_called_once_with(2)
            self.assertEqual(frame["text"].tolist(), ["first", "second"])

    def test_rejects_unknown_split_and_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset split"):
            load_split("pretrain", "unknown")
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    load_split("pretrain", "phase1_train", max_rows=limit)

    def test_error_includes_split_and_original_reason(self):
        with patch("data_exploration.loader.load_dataset", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "pretrain / phase1_train.*offline"):
                load_split("pretrain", "phase1_train")

    def test_empty_preview_preserves_schema(self):
        with patch("data_exploration.loader.load_dataset") as remote:
            remote.return_value.take.return_value = []
            remote.return_value.column_names = ["text", "language"]
            frame = load_split("pretrain", "phase1_train", max_rows=1)
            self.assertTrue(frame.empty)
            self.assertEqual(list(frame), ["text", "language"])


if __name__ == "__main__":
    unittest.main()
