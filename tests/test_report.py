import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from utils.report import run_reports, write_report


class ReportTests(unittest.TestCase):
    def test_writes_utf8_and_overwrites_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested"
            path = write_report("schema", ["مرحبا"], output)
            self.assertIn("مرحبا", path.read_text(encoding="utf-8"))
            write_report("schema", ["updated"], output)
            self.assertNotIn("مرحبا", path.read_text(encoding="utf-8"))
            self.assertEqual(list(output.iterdir()), [path])

    def test_loads_each_split_once_for_multiple_reports(self):
        frame = pd.DataFrame({"text": ["abc"], "language": ["en"]})
        with tempfile.TemporaryDirectory() as directory, patch("utils.report.load_split", return_value=frame) as load:
            result = run_reports({"first": lambda df, subset: "first result",
                                  "second": lambda df, subset: "second result"},
                                 ["--output-dir", directory, "--max-rows", "1"])
            self.assertEqual(result, 0)
            self.assertEqual(load.call_count, 6)
            report = (Path(directory) / "first.txt").read_text(encoding="utf-8")
            self.assertIn("PREVIEW", report)
            self.assertIn("pretrain / phase1_train", report)
            self.assertIn("finetune / eval", report)

    def test_failure_returns_nonzero_and_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as directory, patch("utils.report.load_split", side_effect=RuntimeError("offline")):
            path = Path(directory) / "schema.txt"
            path.write_text("previous complete result", encoding="utf-8")
            self.assertEqual(run_reports({"schema": lambda df, subset: "new"},
                                         ["--output-dir", directory]), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous complete result")

    def test_split_selection_and_output_path(self):
        with tempfile.TemporaryDirectory() as directory, patch("utils.report.load_split", return_value=pd.DataFrame()) as load:
            self.assertEqual(run_reports({"schema": lambda df, subset: "empty"},
                                         ["--output-dir", directory, "--split", "finetune/eval"]), 0)
            load.assert_called_once_with("finetune", "eval", max_rows=None)


if __name__ == "__main__":
    unittest.main()
