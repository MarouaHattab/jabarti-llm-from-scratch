import pandas as pd
from datasets import load_dataset

if __package__:
    from . import config
else:
    import config


def load_split(subset: str, split: str) -> pd.DataFrame:
    """Load one configured dataset split as a pandas DataFrame."""
    if (subset, split) not in config.SPLITS.values():
        raise ValueError(f"Unknown dataset split: {subset} / {split}")
    try:
        dataset = load_dataset(
            config.REPO,
            subset,
            split=split,
            data_files={split: f"{subset}/{split}-*.parquet"},
            cache_dir=str(config.CACHE_DIR),
        )
        return dataset.to_pandas()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {subset} / {split} from {config.REPO}: {exc}"
        ) from exc
