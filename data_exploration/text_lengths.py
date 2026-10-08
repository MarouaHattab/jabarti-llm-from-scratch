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
        lines.append(pooled.astype(float).describe().round(1).to_string())
    else:
        lines.append("No valid text values; character statistics are N/A.")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    """Write text_lengths.txt for the selected dataset splits."""
    available = {f"{subset}/{split}": (label, subset, split)
                 for label, (subset, split) in config.SPLITS.items()}
    parser = argparse.ArgumentParser(description="Save text_lengths.txt for Jabarti dataset splits.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, help="first-N preview per split; omit for full data")
    parser.add_argument("--split", choices=list(available), action="append")
    args = parser.parse_args(argv)
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be a positive integer")
    title = "Text Lengths"
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
        path = args.output_dir / "text_lengths.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
