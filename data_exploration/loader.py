import pandas as pd
from datasets import load_dataset

if __package__:
    from . import config
else:
    import config


def load_split(subset: str, split: str, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load one configured split, or stream its first N rows for a preview."""
    if (subset, split) not in config.SPLITS.values():
        raise ValueError(f"Unknown dataset split: {subset} / {split}")
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("max_rows must be a positive integer")
    revision = getattr(config, "REVISION", "main")
    try:
        dataset = load_dataset(
            config.REPO,
            subset,
            split=split,
            revision=revision,
            data_files={split: f"{subset}/{split}-*.parquet"},
            cache_dir=str(config.CACHE_DIR),
            streaming=max_rows is not None,
        )
        if max_rows is not None:
            records = list(dataset.take(max_rows))
            if not records:
                return pd.DataFrame(columns=dataset.column_names)
            return pd.DataFrame.from_records(records)
        return dataset.to_pandas()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {subset} / {split} from {config.REPO} "
            f"at revision {revision}: {exc}"
        ) from exc
