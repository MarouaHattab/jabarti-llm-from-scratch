import sys

import pandas as pd

if __package__:
    from . import config
    from .loader import load_split
else:
    import config
    from loader import load_split


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """Count language labels using all split rows as the percentage denominator."""
    if "language" not in frame.columns:
        return "Skipped: missing language column."
    if frame.empty:
        return "No rows; language percentages are N/A."
    lines = ["Language distribution (percentage of all rows):"]
    for language, count in frame["language"].value_counts().items():
        lines.append(f"{language} => {count} samples -- {count / len(frame) * 100:.1f}%")
    lines.append(f"Missing language labels: {frame['language'].isnull().sum()}")
    return "\n".join(lines)


def main() -> int:
    """Write language_distribution.txt using the settings in config.py."""
    title = "Language Distribution"
    sections = [title, "=" * len(title), f"Repository: {config.REPO}", ""]
    try:
        for label, (subset, split) in config.SPLITS.items():
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "language_distribution.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
