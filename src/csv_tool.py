
"""
AgriAssist-RX
Phase 4: Agricultural CSV Analysis Tool

Reads local CSV files and calculates:
- Row count
- Column names
- Sum
- Average
- Minimum
- Maximum

Only files inside the project's data folder
are allowed.
"""

import csv
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (PROJECT_ROOT / "data").resolve()

MAX_FILE_SIZE = 2_000_000
MAX_ROWS = 10000


def load_csv(filename):

    # Allow simple CSV file names only.
    # Prevent access to unrelated local files.
    if Path(filename).name != filename:
        raise ValueError(
            "Enter a CSV filename, not a file path."
        )

    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    path = (DATA_DIR / filename).resolve()

    if path.parent != DATA_DIR:
        raise ValueError("Invalid file location.")

    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {filename}"
        )

    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("CSV file is too large.")

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(
    line for line in file if line.strip()
)

        columns = reader.fieldnames

        if not columns:
            raise ValueError("CSV has no columns.")

        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate column names.")

        rows = []

        for row in reader:

            if len(rows) >= MAX_ROWS:
                raise ValueError("Too many rows.")

            if None in row:
                raise ValueError(
                    "Row contains extra values."
                )

            rows.append(row)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    return columns, rows


def numeric_values(rows, column):

    values = []

    for row in rows:

        raw = row[column].strip()

        if not raw:
            continue

        try:
            number = float(raw)
        except ValueError:
            raise ValueError(
                f"Non-numeric value in {column}: {raw}"
            )

        if not (
            float("-inf") < number < float("inf")
        ):
            raise ValueError(
                "Non-finite numbers are not supported."
            )

        values.append(number)

    if not values:
        raise ValueError(
            f"No numeric data in column: {column}"
        )

    return values


def analyze_csv(filename, column):

    columns, rows = load_csv(filename)

    if column not in columns:
        raise ValueError(
            "Column not found. Available: "
            + ", ".join(columns)
        )

    values = numeric_values(rows, column)

    return {
        "filename": filename,
        "column": column,
        "total_rows": len(rows),
        "numeric_values": len(values),
        "sum": sum(values),
        "average": mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():

    print("=" * 55)
    print("AGRIASSIST-RX: CSV ANALYSIS TOOL")
    print("=" * 55)

    filename = input(
        "\nEnter CSV filename: "
    ).strip()

    try:

        columns, rows = load_csv(filename)

        print("\nRows:", len(rows))
        print("Columns:", ", ".join(columns))

        column = input(
            "\nEnter numeric column to analyze: "
        ).strip()

        result = analyze_csv(filename, column)

        print("\n" + "-" * 55)
        print("COLUMN:", result["column"])
        print("SUM:", round(result["sum"], 2))
        print("AVERAGE:", round(result["average"], 2))
        print("MINIMUM:", result["minimum"])
        print("MAXIMUM:", result["maximum"])
        print("-" * 55)

    except (
        ValueError,
        FileNotFoundError,
        OSError
    ) as error:

        print("\nError:", error)


if __name__ == "__main__":
    main()
