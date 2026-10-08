import pandas as pd

from utils.report import run_reports


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


def main(argv: list[str] | None = None) -> int:
    """Generate missing_values.txt for the selected splits."""
    return run_reports({"missing_values": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
