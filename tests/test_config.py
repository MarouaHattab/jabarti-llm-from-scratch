import unittest

from data_exploration.config import CHARS_PER_TOKEN, CONTEXT_LENGTHS, REPO, SPLITS


class ConfigTests(unittest.TestCase):
    def test_preserves_notebook_configuration(self):
        self.assertEqual(REPO, "bakrianoo/jabarti-llm-dataset")
        self.assertEqual(CHARS_PER_TOKEN, 3.5)
        self.assertEqual(CONTEXT_LENGTHS, (512, 1024, 2048))
        self.assertEqual(list(SPLITS.values()), [
            ("pretrain", "phase1_train"), ("pretrain", "phase1_eval"),
            ("pretrain", "phase2_train"), ("pretrain", "phase2_eval"),
            ("finetune", "train"), ("finetune", "eval"),
        ])


if __name__ == "__main__":
    unittest.main()
