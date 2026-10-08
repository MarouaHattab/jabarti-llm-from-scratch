import sys

import pandas as pd

if __package__:
    from . import config
    from .loader import load_split
else:
    import config
    from loader import load_split


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """Report null counts and percentages without treating empty strings as null."""
    rows = len(frame)
    lines = ["Null values per column (percentage of all rows):"]
    nulls = frame.isnull().sum()
    for column, count in nulls.items():
        percentage = f"{count / rows * 100:.1f}%" if rows else "N/A"
        lines.append(f"{column}: {count} ({percentage})")
    if not rows:
        lines.append("No rows; missing-value percentages are N/A.")
    elif not nulls.any():
        lines.append("Columns with missing values: NONE")
    return "\n".join(lines)


def main() -> int:
    """Write missing_values.txt using the settings in config.py."""
    title = "Missing Values"
    sections = [title, "=" * len(title), f"Repository: {config.REPO}", ""]
    try:
        for label, (subset, split) in config.SPLITS.items():
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "missing_values.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
