import sys

import pandas as pd

if __package__:
    from . import config
    from .loader import load_split
else:
    import config
    from loader import load_split


def analyze(frame: pd.DataFrame, subset: str) -> str:
    """List the columns and pandas schema of a loaded dataset split."""
    lines = [f"Columns: {len(frame.columns)}", "Column data types:"]
    lines.extend(f"{column}: {dtype}" for column, dtype in frame.dtypes.items())
    if not len(frame.columns):
        lines.append("No columns available.")
    return "\n".join(lines)


def main() -> int:
    """Write schema.txt using the settings in config.py."""
    title = "Schema"
    sections = [title, "=" * len(title), f"Repository: {config.REPO}", ""]
    try:
        for label, (subset, split) in config.SPLITS.items():
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "schema.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
