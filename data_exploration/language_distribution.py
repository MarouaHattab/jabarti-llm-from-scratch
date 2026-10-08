if not __package__:
    from _bootstrap import prepare_imports
    prepare_imports()

import pandas as pd

from utils.report import run_reports


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


def main(argv: list[str] | None = None) -> int:
    """Generate language_distribution.txt for the selected splits."""
    return run_reports({"language_distribution": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
