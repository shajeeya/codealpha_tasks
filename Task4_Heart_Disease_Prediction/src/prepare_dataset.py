
"""Inspect and validate the UCI Cleveland heart disease dataset."""

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed.cleveland.data"

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    records = []
    missing_values = 0

    with DATA_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for line_number, row in enumerate(reader, start=1):
            if len(row) != len(COLUMN_NAMES):
                raise ValueError(
                    f"Line {line_number}: expected 14 columns, "
                    f"found {len(row)}."
                )

            row = [value.strip() for value in row]
            missing_values += row.count("?")
            records.append(row)

    if not records:
        raise ValueError("The dataset contains no records.")

    target_values = set()

    for line_number, row in enumerate(records, start=1):
        for index, value in enumerate(row):
            if value == "?":
                continue

            try:
                number = float(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid value at line {line_number}, "
                    f"column {COLUMN_NAMES[index]}: {value}"
                ) from error

            if index == len(COLUMN_NAMES) - 1:
                if not number.is_integer() or int(number) not in range(5):
                    raise ValueError(
                        f"Invalid target value on line {line_number}: {value}"
                    )
                target_values.add(int(number))

    print("Dataset inspection completed.")
    print(f"Total records: {len(records)}")
    print(f"Columns per record: {len(COLUMN_NAMES)}")
    print(f"Missing-value markers: {missing_values}")
    print(f"Target values found: {sorted(target_values)}")
    print("Column names:")
    print(", ".join(COLUMN_NAMES))


if __name__ == "__main__":
    main()
