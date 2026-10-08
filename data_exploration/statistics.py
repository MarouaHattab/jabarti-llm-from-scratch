if not __package__:
    from _bootstrap import prepare_imports
    prepare_imports()

import pandas as pd

from utils.report import run_reports


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """Report the notebook's sample count and the loaded table shape."""
    return f"Samples: {len(frame)}\nColumns: {len(frame.columns)}"


def main(argv: list[str] | None = None) -> int:
    """Generate statistics.txt for the selected splits."""
    return run_reports({"statistics": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
