from data_exploration import (
    duplicates,
    language_distribution,
    missing_values,
    schema,
    statistics,
    text_lengths,
    token_estimation,
)
from utils.report import run_reports


def main(argv: list[str] | None = None) -> int:
    """Generate all seven reports while loading each split only once."""
    analyses = {
        "schema": schema.analyze,
        "statistics": statistics.analyze,
        "missing_values": missing_values.analyze,
        "language_distribution": language_distribution.analyze,
        "text_lengths": text_lengths.analyze,
        "token_estimation": token_estimation.analyze,
        "duplicates": duplicates.analyze,
    }
    return run_reports(analyses, argv)


if __name__ == "__main__":
    raise SystemExit(main())
