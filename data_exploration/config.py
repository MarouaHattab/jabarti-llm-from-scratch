"""Notebook constants and project-relative output paths.

Requires Python 3.11+. Install with ``python -m pip install -r requirements.txt``.
Run every report with ``python -m data_exploration.run`` or run an individual
module, for example ``python -m data_exploration.schema``. Each command accepts
``--help``. Tests run with ``python -m unittest discover -s tests -v``.

All results are saved as UTF-8 TXT files under outputs/data_exploration.
The original notebook and its saved results remain in Notebooks/ unchanged.
"""

from pathlib import Path

REPO = "bakrianoo/jabarti-llm-dataset"
# The notebook used the default upstream revision. Pin the revision audited
# during refactoring so subsequent runs use the same dataset contents.
REVISION = "7b7aeaaff76fdc254a9d245146c9d8a624bbd552"
SPLITS = {
    "pretrain / phase1_train": ("pretrain", "phase1_train"),
    "pretrain / phase1_eval": ("pretrain", "phase1_eval"),
    "pretrain / phase2_train": ("pretrain", "phase2_train"),
    "pretrain / phase2_eval": ("pretrain", "phase2_eval"),
    "finetune / train": ("finetune", "train"),
    "finetune / eval": ("finetune", "eval"),
}
TEXT_COLUMNS = {"pretrain": ("text",), "finetune": ("question", "answer")}
CHARS_PER_TOKEN = 3.5
CONTEXT_LENGTHS = (512, 1024, 2048)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_exploration"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
