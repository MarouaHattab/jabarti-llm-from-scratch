"""Load validated exploration settings using Python's built-in TOML parser."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs/data_config.toml"


@dataclass(frozen=True)
class SplitConfig:
    """Columns and source location for one independently explored split."""

    subset: str
    name: str
    text_columns: tuple[str, ...]
    duplicate_keys: tuple[str, ...] = ()
    local_path: Path | None = None

    @property
    def label(self) -> str:
        """Return the same split label used by the original notebook."""
        return f"{self.subset} / {self.name}"


@dataclass(frozen=True)
class ExplorationConfig:
    """Immutable settings shared by loading, analysis, and reporting."""

    repo: str
    revision: str
    output_dir: Path
    cache_dir: Path
    splits: tuple[SplitConfig, ...]
    chars_per_token: float = 3.5
    context_lengths: tuple[int, ...] = (512, 1024, 2048)
    max_rows: int | None = None


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strings(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a list of nonempty column names")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)


def _resolve_path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    return (base / Path(value).expanduser()).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> ExplorationConfig:
    """Read TOML and resolve paths relative to the configuration file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc

    dataset = raw.get("dataset", {})
    paths = raw.get("paths", {})
    analysis = raw.get("analysis", {})
    for name, table in (("dataset", dataset), ("paths", paths), ("analysis", analysis)):
        if not isinstance(table, dict):
            raise ValueError(f"{name} must be a TOML table")
    repo = dataset.get("repo", "bakrianoo/jabarti-llm-dataset")
    revision = dataset.get("revision", "7b7aeaaff76fdc254a9d245146c9d8a624bbd552")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError("dataset.repo must be a nonempty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("dataset.revision must be a nonempty revision")
    chars_per_token = analysis.get("chars_per_token", 3.5)
    if (isinstance(chars_per_token, bool) or not isinstance(chars_per_token, (int, float))
            or not math.isfinite(chars_per_token) or chars_per_token <= 0):
        raise ValueError("chars_per_token must be a finite positive number")
    contexts = analysis.get("context_lengths", [512, 1024, 2048])
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("context_lengths must be a nonempty list")
    context_lengths = tuple(_positive_int(value, "context_lengths") for value in contexts)
    if len(set(context_lengths)) != len(context_lengths):
        raise ValueError("context_lengths must not contain duplicates")
    max_rows = analysis.get("max_rows")
    if max_rows is not None:
        max_rows = _positive_int(max_rows, "max_rows")

    entries = raw.get("splits", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Configure at least one [[splits]] entry")
    splits = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each split must be a TOML table")
        subset, name = entry.get("subset"), entry.get("name")
        if any(not isinstance(value, str) or not value.strip() for value in (subset, name)):
            raise ValueError("Each split requires nonempty subset and name strings")
        if any(char in subset + name for char in "/*?[]\\"):
            raise ValueError("Split subset and name cannot contain path or glob characters")
        if (subset, name) in seen:
            raise ValueError(f"Duplicate split: {subset} / {name}")
        seen.add((subset, name))
        local_path = entry.get("local_path")
        splits.append(SplitConfig(
            subset=subset,
            name=name,
            text_columns=_strings(entry.get("text_columns"), "text_columns"),
            duplicate_keys=_strings(entry.get("duplicate_keys", []), "duplicate_keys", allow_empty=True),
            local_path=None if local_path is None else _resolve_path(path.parent, local_path, "local_path"),
        ))

    return ExplorationConfig(
        repo=repo,
        revision=revision,
        output_dir=_resolve_path(path.parent, paths.get("output_dir", "../outputs/data_exploration"), "output_dir"),
        cache_dir=_resolve_path(path.parent, paths.get("cache_dir", "../data/cache"), "cache_dir"),
        splits=tuple(splits),
        chars_per_token=float(chars_per_token),
        context_lengths=context_lengths,
        max_rows=max_rows,
    )
