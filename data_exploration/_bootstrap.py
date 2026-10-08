import sys
from pathlib import Path


def prepare_imports() -> None:
    """Resolve shared imports when an analysis is executed by filename."""
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
