import sys
import math

import pandas as pd

if __package__:
    from . import config
    from .loader import load_split
    from .text_lengths import character_lengths
else:
    import config
    from loader import load_split
    from text_lengths import character_lengths


def estimate_tokens(frame: pd.DataFrame, subset: str) -> pd.Series:
    if (not isinstance(config.CHARS_PER_TOKEN, (int, float))
            or isinstance(config.CHARS_PER_TOKEN, bool)
            or not math.isfinite(config.CHARS_PER_TOKEN) or config.CHARS_PER_TOKEN <= 0):
        raise ValueError("CHARS_PER_TOKEN must be a finite positive number")
    lengths = character_lengths(frame, subset)
    sample_chars = lengths.sum(axis=1, min_count=len(lengths.columns)).dropna()
    return (sample_chars / config.CHARS_PER_TOKEN).astype("int64")


def analyze(frame: pd.DataFrame, subset: str) -> str:
    tokens = estimate_tokens(frame, subset)
    lines = ["Token counts are estimates, not exact tokenizer results.",
             f"Formula: int(sample character count / {config.CHARS_PER_TOKEN})",
             "Finetune sample characters = question characters + answer characters.",
             "Statistics cover the split once, independently of language labels.",
             f"Valid samples: {len(tokens)}", f"Samples excluded: {len(frame) - len(tokens)}"]
    if tokens.empty:
        lines.append("No valid samples; estimated-token statistics and coverage are N/A.")
        return "\n".join(lines)
    lines.extend(["Estimated tokens per sample:", tokens.astype(float).describe().round(1).to_string(),
                  "Context coverage uses valid samples; no data is truncated or removed."])
    for context in config.CONTEXT_LENGTHS:
        if type(context) is not int or context <= 0:
            raise ValueError("CONTEXT_LENGTHS must contain positive integers")
        dropped = (tokens > context).mean() * 100
        lines.append(f"max_length={context} | kept: {100 - dropped:.1f}% | dropped: {dropped:.1f}%")
    return "\n".join(lines)


def main() -> int:
    title = "Token Estimation"
    sections = [title, "=" * len(title), f"Repository: {config.REPO}", ""]
    try:
        for label, (subset, split) in config.SPLITS.items():
            print(f"Exploring {label}", flush=True)
            frame = load_split(subset, split)
            sections.append(f"{label}\n{'-' * len(label)}\n{analyze(frame, subset)}\n")
            del frame
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "token_estimation.txt"
        path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {path}")
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Exploration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
