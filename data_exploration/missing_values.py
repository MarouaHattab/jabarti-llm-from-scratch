import argparse
import sys
from pathlib import Path

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

def main(argv: list[str] | None = None) -> int:
    """Write missing_values.txt for the selected dataset splits."""
    available = {f"{subset}/{split}": (label, subset, split)
                 for label, (subset, split) in config.SPLITS.items()}
    parser = argparse.ArgumentParser(description="Save missing_values.txt for Jabarti dataset splits.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, help="first-N preview per split; omit for full data")
    parser.add_argument("--split", choices=list(available), action="append")
    args = parser.parse_args(argv)
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be a positive integer")
    title = "Missing Values"
    scope = ("FULL configured splits" if args.max_rows is None else
             f"PREVIEW first {args.max_rows} rows per split; not full-dataset statistics")
    sections = [title, "=" * len(title), f"Repository: {config.REPO}",
                f"Revision: {getattr(config, 'REVISION', 'main (upstream default)')}",
                f"Scope: {scope}", ""]
    try:
        for key in dict.fromkeys(args.split or available):
            label, subset, split = available[key]
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split, max_rows=args.max_rows)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "missing_values.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
