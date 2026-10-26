
"""Basic tests for the Cleveland heart disease dataset."""

import csv

from prepare_dataset import COLUMN_NAMES, DATA_PATH


def test_dataset_file_exists():
    """Check that the dataset file exists."""
    assert DATA_PATH.is_file()


def test_dataset_has_expected_records():
    """Check that the dataset contains 303 records."""
    with DATA_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 303


def test_dataset_has_14_columns():
    """Check that every record has 14 columns."""
    with DATA_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(COLUMN_NAMES) == 14
    assert all(len(row) == 14 for row in rows)


def test_dataset_contains_missing_value_markers():
    """Check that missing values are present in the raw dataset."""
    with DATA_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert sum(value.strip() == "?" for row in rows for value in row) == 6
