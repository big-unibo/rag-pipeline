from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("resources/datasets/novel/questions.json")
DEFAULT_OUTPUT = Path("resources/datasets/novel/questions.csv")
REQUIRED_FIELDS = ("id", "question", "answer")
OUTPUT_FIELDS = ("id", "text", "ground_truth")
MAPPED_FIELDS = frozenset((*REQUIRED_FIELDS, *OUTPUT_FIELDS))


def convert_questions(input_path: Path, output_path: Path) -> int:
    """Convert question records from JSON to a semicolon-delimited CSV."""
    with input_path.open(encoding="utf-8") as input_file:
        records: Any = json.load(input_file)

    if not isinstance(records, list):
        raise ValueError("The input JSON must contain a list of question objects.")

    extra_fields: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Question record {index} must be a JSON object.")
        missing_fields = [field for field in REQUIRED_FIELDS if field not in record]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Question record {index} is missing required field(s): {missing}.")
        for field in record:
            if field not in MAPPED_FIELDS and field not in extra_fields:
                extra_fields.append(field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[*OUTPUT_FIELDS, *extra_fields],
            delimiter=";",
        )
        writer.writeheader()
        for record in records:
            row = {
                "id": record["id"],
                "text": record["question"],
                "ground_truth": record["answer"],
            }
            row.update(
                {
                    field: record[field]
                    for field in extra_fields
                    if field in record
                }
            )
            writer.writerow(
                {
                    field: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else "" if value is None else value
                    for field, value in row.items()
                }
            )

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert question JSON dataset to a semicolon-delimited CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input questions JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output questions CSV.")
    args = parser.parse_args()

    count = convert_questions(args.input, args.output)
    print(f"Converted {count} questions: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()