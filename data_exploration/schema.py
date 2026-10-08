if not __package__:
    from _bootstrap import prepare_imports
    prepare_imports()

import pandas as pd

from utils.report import run_reports


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """List the columns and pandas schema of a loaded dataset split."""
    lines = [f"Columns: {len(frame.columns)}", "Column data types:"]
    lines.extend(f"{column}: {dtype}" for column, dtype in frame.dtypes.items())
    if not len(frame.columns):
        lines.append("No columns available.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Generate schema.txt for the selected splits."""
    return run_reports({"schema": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
