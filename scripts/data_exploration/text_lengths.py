import sys

import pandas as pd

if __package__:
    from . import config
    from .loader import load_split
else:
    import config
    from loader import load_split


def character_lengths(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
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
        lines.append(pooled.astype(float).describe().round(1).to_string())
    else:
        lines.append("No valid text values; character statistics are N/A.")
    return "\n".join(lines)


def main() -> int:
    title = "Text Lengths"
    sections = [title, "=" * len(title), f"Repository: {config.REPO}", ""]
    try:
        for label, (subset, split) in config.SPLITS.items():
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "text_lengths.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
