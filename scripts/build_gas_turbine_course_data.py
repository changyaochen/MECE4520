#!/usr/bin/env python3
"""Build the documented MECE 4520 gas-turbine course dataset.

The raw UCI data are intentionally not committed. This script accepts either
the original concatenated CSV (in chronological annual order) or a directory
containing the five annual UCI CSV files.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


SOURCE_COLUMNS = (
    "AT",
    "AP",
    "AH",
    "AFDP",
    "GTEP",
    "TIT",
    "TAT",
    "TEY",
    "CDP",
    "CO",
    "NOX",
)
CAMPAIGN_ROWS = {
    2011: 7411,
    2012: 7628,
    2013: 7152,
    2014: 7158,
    2015: 7384,
}
RANDOM_SEED = 4520
AH_MISSING_FRACTION = 0.03
AFDP_BLOCK_LENGTH = 24
AFDP_BLOCKS_PER_CAMPAIGN = 2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(
                f"{path} must contain exactly these columns: {', '.join(SOURCE_COLUMNS)}"
            )
        rows = list(reader)

    if any(not row[column] for row in rows for column in SOURCE_COLUMNS):
        raise ValueError(f"{path} is expected to be the clean UCI source data.")
    return rows


def read_concatenated_source(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    expected_rows = sum(CAMPAIGN_ROWS.values())
    if len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows; expected {expected_rows}.")

    start = 0
    labelled_rows: list[dict[str, str]] = []
    for year, count in CAMPAIGN_ROWS.items():
        for row in rows[start : start + count]:
            labelled_rows.append({"campaign_year": str(year), **row})
        start += count
    return labelled_rows


def read_annual_sources(directory: Path) -> list[dict[str, str]]:
    labelled_rows: list[dict[str, str]] = []
    for year, expected_count in CAMPAIGN_ROWS.items():
        path = directory / f"gt_{year}.csv"
        rows = read_csv(path)
        if len(rows) != expected_count:
            raise ValueError(
                f"{path} has {len(rows)} rows; expected {expected_count} for {year}."
            )
        labelled_rows.extend({"campaign_year": str(year), **row} for row in rows)
    return labelled_rows


def add_documented_missingness(rows: list[dict[str, str]]) -> None:
    """Blank selected feature values without changing any measured target."""

    rng = random.Random(RANDOM_SEED)

    # Isolated missing humidity readings: 3% of all records.
    for index in rng.sample(range(len(rows)), round(AH_MISSING_FRACTION * len(rows))):
        rows[index]["AH"] = ""

    # Two non-overlapping 24-record AFDP outages in each annual campaign.
    for year in CAMPAIGN_ROWS:
        campaign_indices = [
            index for index, row in enumerate(rows) if row["campaign_year"] == str(year)
        ]
        candidate_starts = range(0, len(campaign_indices) - AFDP_BLOCK_LENGTH + 1, AFDP_BLOCK_LENGTH)
        for start in rng.sample(list(candidate_starts), AFDP_BLOCKS_PER_CAMPAIGN):
            for index in campaign_indices[start : start + AFDP_BLOCK_LENGTH]:
                rows[index]["AFDP"] = ""


def write_csv(rows: list[dict[str, str]], output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("campaign_year", *SOURCE_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=Path,
        help="Clean, concatenated UCI CSV in chronological annual order.",
    )
    source_group.add_argument(
        "--uci-directory",
        type=Path,
        help="Directory containing gt_2011.csv through gt_2015.csv from UCI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/data/gas-turbine-course.csv"),
        help="Destination for the derived course CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = (
        read_concatenated_source(args.source)
        if args.source is not None
        else read_annual_sources(args.uci_directory)
    )
    add_documented_missingness(rows)
    write_csv(rows, args.output, args.force)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
