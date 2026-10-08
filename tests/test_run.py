import ast
import hashlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_exploration import config

MODULES = ("schema", "statistics", "missing_values", "language_distribution",
           "text_lengths", "token_estimation", "duplicates")


def sample_frame(subset: str) -> pd.DataFrame:
    if subset == "pretrain":
        return pd.DataFrame({"text": ["مرحبا", "hello", "hello"],
                             "language": ["ar", "en", "en"],
                             "article_id": ["1", "2", "2"], "chunk_index": [0, 0, 0]})
    return pd.DataFrame({"question": ["من؟", "who?"], "answer": ["أنا", "me"],
                         "language": ["ar", "en"]})


class RunTests(unittest.TestCase):
    def test_constants_match_original_notebook(self):
        notebook = json.loads((config.PROJECT_ROOT / "Notebooks/datasets_exploration.ipynb").read_text(encoding="utf-8"))
        original = {}
        for index in (1, 3):
            tree = ast.parse("".join(notebook["cells"][index]["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ("REPO", "splits", "CHARS_PER_TOKEN", "CONTEXT_LENGHTS"):
                            original[target.id] = ast.literal_eval(node.value)
        self.assertEqual(config.REPO, original["REPO"])
        self.assertEqual(config.SPLITS, original["splits"])
        self.assertEqual(config.CHARS_PER_TOKEN, original["CHARS_PER_TOKEN"])
        self.assertEqual(config.CONTEXT_LENGTHS, tuple(original["CONTEXT_LENGHTS"]))

    def test_full_runner_loads_each_split_once_and_saves_seven_reports(self):
        from data_exploration.run import main
        notebook = config.PROJECT_ROOT / "Notebooks/datasets_exploration.ipynb"
        before = hashlib.sha256(notebook.read_bytes()).digest()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
            "utils.report.load_split", side_effect=lambda subset, split, **kwargs: sample_frame(subset)
        ) as load:
            self.assertEqual(main(["--output-dir", directory]), 0)
            self.assertEqual(load.call_count, 6)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                             sorted(f"{name}.txt" for name in MODULES))
            text = (Path(directory) / "text_lengths.txt").read_text(encoding="utf-8")
            self.assertEqual(text.count("Character length distribution"), 6)
            self.assertIn("FULL", text)
        self.assertEqual(hashlib.sha256(notebook.read_bytes()).digest(), before)

    def test_independent_modules_write_only_their_own_report(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
                "utils.report.load_split", return_value=sample_frame("pretrain")
            ):
                module = importlib.import_module(f"data_exploration.{name}")
                self.assertEqual(module.main(["--output-dir", directory, "--split", "pretrain/phase1_eval"]), 0)
                self.assertEqual([path.name for path in Path(directory).iterdir()], [f"{name}.txt"])

    def test_all_reports_handle_empty_and_missing_columns(self):
        from data_exploration.run import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
            "utils.report.load_split", return_value=pd.DataFrame()
        ):
            self.assertEqual(main(["--output-dir", directory]), 0)
            for name in MODULES:
                self.assertTrue((Path(directory) / f"{name}.txt").is_file())

    def test_invalid_output_path_fails_clearly(self):
        from data_exploration.run import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors, patch(
            "utils.report.load_split", return_value=sample_frame("pretrain")
        ):
            path = Path(directory) / "file.txt"
            path.write_text("existing file", encoding="utf-8")
            self.assertEqual(main(["--output-dir", str(path)]), 1)
            self.assertIn("Exploration failed", errors.getvalue())

    def test_module_command_line_imports_without_sys_path_changes(self):
        for name in (*MODULES, "run"):
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, "-m", f"data_exploration.{name}", "--help"],
                                        cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--max-rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
