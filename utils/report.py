import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from data_exploration import config
from data_exploration.loader import load_split

Analysis = Callable[[pd.DataFrame, str], str]


def describe(values: pd.Series) -> str:
    """Format the notebook's descriptive statistics to one decimal place."""
    return values.astype(float).describe().round(1).to_string()


def write_report(
    name: str,
    sections: Iterable[str],
    output_dir: Path = config.OUTPUT_DIR,
    *,
    max_rows: int | None = None,
) -> Path:
    """Save a deterministic UTF-8 report, replacing its previous contents."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.txt"
    scope = ("FULL configured splits" if max_rows is None else
             f"PREVIEW first {max_rows} rows per split; not full-dataset statistics")
    heading = name.replace("_", " ").title()
    content = [heading, "=" * len(heading), f"Repository: {config.REPO}",
               f"Revision: {getattr(config, 'REVISION', 'main (upstream default)')}",
               f"Scope: {scope}", "", *sections]
    path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")
    return path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def run_reports(analyses: dict[str, Analysis], argv: list[str] | None = None) -> int:
    """Run selected analyses, loading one split at a time and reusing its frame."""
    available = {f"{subset}/{split}": (label, subset, split)
                 for label, (subset, split) in config.SPLITS.items()}
    parser = argparse.ArgumentParser(
        description="Explore Jabarti splits and save separate UTF-8 TXT reports.",
        epilog=("From data_exploration, execute python <script_name>.py with Python 3.11+. "
                "Configure the dataset and constants in config.py. Install dependencies "
                "from the project root using python -m pip install -r requirements.txt. The original "
                "notebook is preserved in Notebooks/datasets_exploration.ipynb."),
    )
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR,
                        help="report directory (default: outputs/data_exploration under project root)")
    parser.add_argument("--max-rows", type=_positive_int,
                        help="deterministic first-N preview per split; omit for complete splits")
    parser.add_argument("--split", choices=list(available), action="append",
                        help="restrict to one split; repeat for several splits")
    args = parser.parse_args(argv)
    selected = list(dict.fromkeys(args.split or available))
    sections: dict[str, list[str]] = {name: [] for name in analyses}
    try:
        for key in selected:
            label, subset, split = available[key]
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split, max_rows=args.max_rows)
            for name, analyze in analyses.items():
                sections[name].append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        for name, parts in sections.items():
            path = write_report(name, parts, args.output_dir, max_rows=args.max_rows)
            print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0
