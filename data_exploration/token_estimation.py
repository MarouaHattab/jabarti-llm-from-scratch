if not __package__:
    from _bootstrap import prepare_imports
    prepare_imports()

import math

import pandas as pd

from data_exploration import config
from data_exploration.text_lengths import character_lengths
from utils.report import describe, run_reports


def estimate_tokens(frame: pd.DataFrame, subset: str) -> pd.Series:
    """Estimate tokens by summing sample characters, dividing, then truncating."""
    if (not isinstance(config.CHARS_PER_TOKEN, (int, float))
            or isinstance(config.CHARS_PER_TOKEN, bool)
            or not math.isfinite(config.CHARS_PER_TOKEN) or config.CHARS_PER_TOKEN <= 0):
        raise ValueError("CHARS_PER_TOKEN must be a finite positive number")
    lengths = character_lengths(frame, subset)
    sample_chars = lengths.sum(axis=1, min_count=len(lengths.columns)).dropna()
    return (sample_chars / config.CHARS_PER_TOKEN).astype("int64")


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """Describe approximate token counts and inclusive context coverage."""
    tokens = estimate_tokens(frame, subset)
    lines = ["Token counts are estimates, not exact tokenizer results.",
             f"Formula: int(sample character count / {config.CHARS_PER_TOKEN})",
             "Finetune sample characters = question characters + answer characters.",
             "Statistics cover the split once, independently of language labels.",
             f"Valid samples: {len(tokens)}", f"Samples excluded: {len(frame) - len(tokens)}"]
    if tokens.empty:
        lines.append("No valid samples; estimated-token statistics and coverage are N/A.")
        return "\n".join(lines)
    lines.extend(["Estimated tokens per sample:", describe(tokens),
                  "Context coverage uses valid samples; no data is truncated or removed."])
    for context in config.CONTEXT_LENGTHS:
        if type(context) is not int or context <= 0:
            raise ValueError("CONTEXT_LENGTHS must contain positive integers")
        dropped = (tokens > context).mean() * 100
        lines.append(f"max_length={context} | kept: {100 - dropped:.1f}% | dropped: {dropped:.1f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Generate token_estimation.txt for the selected splits."""
    return run_reports({"token_estimation": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
