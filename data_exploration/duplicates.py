import sys

import pandas as pd

if __package__:
    from . import config
    from .loader import load_split
else:
    import config
    from loader import load_split


def analyze(frame: pd.DataFrame, subset: str) -> str:
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


def main() -> int:
    title = "Duplicates"
    sections = [title, "=" * len(title), f"Repository: {config.REPO}", ""]
    try:
        for label, (subset, split) in config.SPLITS.items():
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "duplicates.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
