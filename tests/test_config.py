"""Configuration checks independent of network and dataset availability."""

import tempfile
import unittest
from pathlib import Path

from src.config import DEFAULT_CONFIG, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_preserve_notebook_splits_and_parameters(self):
        config = load_config()
        self.assertEqual(config.repo, "bakrianoo/jabarti-llm-dataset")
        self.assertEqual(len(config.revision), 40)
        self.assertEqual(config.chars_per_token, 3.5)
        self.assertEqual(config.context_lengths, (512, 1024, 2048))
        self.assertEqual(
            [(split.subset, split.name) for split in config.splits],
            [("pretrain", "phase1_train"), ("pretrain", "phase1_eval"),
             ("pretrain", "phase2_train"), ("pretrain", "phase2_eval"),
             ("finetune", "train"), ("finetune", "eval")],
        )
        self.assertEqual(config.output_dir, DEFAULT_CONFIG.parent.parent / "outputs/data_exploration")

    def test_paths_are_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "custom.toml"
            config_path.write_text(
                '[paths]\noutput_dir="reports"\ncache_dir="cache"\n'
                '[[splits]]\nsubset="pretrain"\nname="test"\n'
                'text_columns=["text"]\nlocal_path="sample.jsonl"\n',
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.output_dir, Path(directory) / "reports")
            self.assertEqual(config.splits[0].local_path, Path(directory) / "sample.jsonl")

    def test_missing_file_is_actionable(self):
        with self.assertRaisesRegex(FileNotFoundError, "Configuration file"):
            load_config(Path("nonexistent-config.toml"))

    def test_rejects_invalid_analysis_parameters_and_splits(self):
        cases = [
            '[analysis]\nchars_per_token=0\n',
            '[analysis]\nchars_per_token=nan\n',
            '[analysis]\nmax_rows=0\n',
            '[analysis]\nmax_rows=true\n',
            '[analysis]\ncontext_lengths=[512,-1]\n',
            '[analysis]\ncontext_lengths=[]\n',
            '[[splits]]\nsubset="pretrain"\nname="test"\ntext_columns=[]\n',
            '[[splits]]\nsubset="pretrain"\nname="test"\ntext_columns=["text"]\n'
            '[[splits]]\nsubset="pretrain"\nname="test"\ntext_columns=["text"]\n',
        ]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.toml"
            for content in cases:
                with self.subTest(content=content):
                    config_path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path)


if __name__ == "__main__":
    unittest.main()
