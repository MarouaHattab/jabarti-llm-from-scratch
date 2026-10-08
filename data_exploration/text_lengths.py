if not __package__:
    from _bootstrap import prepare_imports
    prepare_imports()

import pandas as pd

from data_exploration import config
from utils.report import describe, run_reports


def character_lengths(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Measure strings without coercing missing or invalid values into text."""
    columns = config.TEXT_COLUMNS[subset]
    lengths = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            lengths[column] = frame[column].map(
                lambda value: len(value) if isinstance(value, str) else float("nan")
            ).astype(float)
        else:
            lengths[column] = float("nan")
    return lengths


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """Describe text lengths; pool question and answer lengths for finetune."""
    columns = config.TEXT_COLUMNS[subset]
    missing = [column for column in columns if column not in frame.columns]
    lengths = character_lengths(frame, subset)
    pooled = pd.concat([lengths[column] for column in columns], ignore_index=True)
    lines = [f"Character length distribution for: {', '.join(columns)}",
             "Each text field is a separate observation; questions and answers are pooled.",
             "Statistics cover the split once, independently of language labels."]
    if missing:
        lines.append(f"Missing columns: {', '.join(missing)}")
    lines.append(f"Missing or non-string text values excluded: {pooled.isna().sum()}")
    if pooled.notna().any():
        lines.append(describe(pooled))
    else:
        lines.append("No valid text values; character statistics are N/A.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Generate text_lengths.txt for the selected splits."""
    return run_reports({"text_lengths": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
