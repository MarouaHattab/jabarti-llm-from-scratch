import unittest

import pandas as pd


class TextLengthTests(unittest.TestCase):
    def test_pretrain_matches_notebook_formula(self):
        from data_exploration.text_lengths import analyze, character_lengths
        frame = pd.DataFrame({"text": ["مرحبا", "hello world", "abc"]})
        expected = frame["text"].str.len().describe().round(1).to_string()
        self.assertIn(expected, analyze(frame, "pretrain"))
        self.assertEqual(character_lengths(frame, "pretrain")["text"].tolist(), [5.0, 11.0, 3.0])

    def test_finetune_pools_question_and_answer_lengths(self):
        from data_exploration.text_lengths import analyze
        frame = pd.DataFrame({"question": ["abc", "ab"], "answer": ["123456", "1234"]})
        expected = pd.concat([frame["question"].str.len(), frame["answer"].str.len()],
                             ignore_index=True).describe().round(1).to_string()
        self.assertIn(expected, analyze(frame, "finetune"))

    def test_invalid_values_are_not_converted_to_text(self):
        from data_exploration.text_lengths import analyze, character_lengths
        frame = pd.DataFrame({"text": ["abc", None, 123]})
        lengths = character_lengths(frame, "pretrain")["text"]
        self.assertEqual(lengths.notna().sum(), 1)
        self.assertIn("Missing or non-string text values excluded: 2", analyze(frame, "pretrain"))
        self.assertIn("Missing columns: answer", analyze(pd.DataFrame({"question": ["abc"]}), "finetune"))



if __name__ == "__main__":
    unittest.main()
