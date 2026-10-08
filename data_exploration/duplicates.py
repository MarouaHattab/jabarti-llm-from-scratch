if not __package__:
    from _bootstrap import prepare_imports
    prepare_imports()

import pandas as pd

from utils.report import run_reports


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """Count duplicate occurrences after the first, matching pandas.duplicated."""
    lines = ["Duplicates exclude the first occurrence; percentages use all split rows.",
             "Checks compare exact stored values without normalization."]
    checks = [("Duplicate text records", ["text"]),
              ("Duplicate (article_id, chunk_index) pairs", ["article_id", "chunk_index"])]
    for label, columns in checks:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            lines.append(f"Skipped {label.lower()}: missing {', '.join(missing)}.")
            continue
        count = int(frame.duplicated(subset=columns).sum())
        percentage = f"{count / len(frame) * 100:.1f}%" if len(frame) else "N/A"
        lines.append(f"{label}: {count} ({percentage})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Generate duplicates.txt for splits with supported columns."""
    return run_reports({"duplicates": analyze}, argv)


if __name__ == "__main__":
    raise SystemExit(main())
