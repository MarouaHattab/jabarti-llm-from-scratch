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


class ScriptTests(unittest.TestCase):
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

    def test_each_script_loads_each_split_once(self):
        notebook = config.PROJECT_ROOT / "Notebooks/datasets_exploration.ipynb"
        before = hashlib.sha256(notebook.read_bytes()).digest()
        for name in MODULES:
            module = importlib.import_module(f"data_exploration.{name}")
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
                f"data_exploration.{name}.load_split", side_effect=lambda subset, split, **kwargs: sample_frame(subset)
            ) as load:
                self.assertEqual(module.main(["--output-dir", directory]), 0)
                self.assertEqual(load.call_count, 6)
                path = Path(directory) / f"{name}.txt"
                self.assertEqual(list(Path(directory).iterdir()), [path])
                text = path.read_text(encoding="utf-8")
                self.assertIn("FULL", text)
                if name == "text_lengths":
                    self.assertEqual(text.count("Character length distribution"), 6)
        self.assertEqual(hashlib.sha256(notebook.read_bytes()).digest(), before)

    def test_independent_modules_write_only_their_own_report(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
                f"data_exploration.{name}.load_split", return_value=sample_frame("pretrain")
            ):
                module = importlib.import_module(f"data_exploration.{name}")
                self.assertEqual(module.main(["--output-dir", directory, "--split", "pretrain/phase1_eval"]), 0)
                self.assertEqual([path.name for path in Path(directory).iterdir()], [f"{name}.txt"])

    def test_all_reports_handle_empty_and_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for name in MODULES:
                module = importlib.import_module(f"data_exploration.{name}")
                with patch.object(module, "load_split", return_value=pd.DataFrame()):
                    self.assertEqual(module.main(["--output-dir", directory]), 0)
                self.assertTrue((Path(directory) / f"{name}.txt").is_file())

    def test_invalid_output_path_fails_clearly(self):
        from data_exploration.schema import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors, patch(
            "data_exploration.schema.load_split", return_value=sample_frame("pretrain")
        ):
            path = Path(directory) / "file.txt"
            path.write_text("existing file", encoding="utf-8")
            self.assertEqual(main(["--output-dir", str(path)]), 1)
            self.assertIn("Exploration failed", errors.getvalue())

    def test_module_command_line_imports_without_sys_path_changes(self):
        for name in MODULES:
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, "-m", f"data_exploration.{name}", "--help"],
                                        cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--max-rows", result.stdout)

    def test_direct_script_command_from_exploration_directory(self):
        for name in MODULES:
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, f"{name}.py", "--help"],
                                        cwd=config.PROJECT_ROOT / "data_exploration",
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--max-rows", result.stdout)

    def test_runner_is_removed(self):
        self.assertFalse((config.PROJECT_ROOT / "data_exploration/run.py").exists())

    def test_no_bootstrap_or_import_path_changes(self):
        directory = config.PROJECT_ROOT / "data_exploration"
        self.assertFalse((directory / "_bootstrap.py").exists())
        self.assertFalse((directory / "utils").exists())
        self.assertFalse((config.PROJECT_ROOT / "utils").exists())
        for path in directory.rglob("*.py"):
            self.assertNotIn("sys.path", path.read_text(encoding="utf-8"), str(path))

    def test_reports_overwrite_in_place_and_preserve_utf8(self):
        from data_exploration.language_distribution import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
            "data_exploration.language_distribution.load_split",
            return_value=pd.DataFrame({"language": ["العربية"]}),
        ):
            args = ["--output-dir", directory, "--split", "finetune/eval"]
            self.assertEqual(main(args), 0)
            path = Path(directory) / "language_distribution.txt"
            self.assertIn("العربية", path.read_text(encoding="utf-8"))
            path.write_text("old contents", encoding="utf-8")
            self.assertEqual(main(args), 0)
            self.assertNotIn("old contents", path.read_text(encoding="utf-8"))
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_load_error_keeps_previous_report(self):
        from data_exploration.schema import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch(
            "data_exploration.schema.load_split", side_effect=RuntimeError("offline"),
        ):
            path = Path(directory) / "schema.txt"
            path.write_text("previous complete report", encoding="utf-8")
            self.assertEqual(main(["--output-dir", directory]), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous complete report")


if __name__ == "__main__":
    unittest.main()
