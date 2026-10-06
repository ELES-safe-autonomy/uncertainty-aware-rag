"""Combine per-drug question files in eval/raw/ into one eval/questions.csv."""

import csv
from collections import Counter
from pathlib import Path

RAW_DIR = Path("eval/raw")
OUT = Path("eval/questions.csv")
HEADER = ["id", "drug", "question", "answer", "type", "label_section", "evidence_quote"]


def main():
    rows, seen_ids = [], set()
    files = sorted(RAW_DIR.glob("*.csv"))

    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as f:
            for line_no, row in enumerate(csv.reader(f), start=1):
                if not row or row == HEADER:
                    continue
                if len(row) != len(HEADER):
                    print(f"SKIPPED {path.name} line {line_no}: expected 7 columns, got {len(row)}")
                    continue
                if row[0] in seen_ids:
                    print(f"SKIPPED {path.name} line {line_no}: duplicate id {row[0]}")
                    continue
                seen_ids.add(row[0])
                rows.append(row)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} questions from {len(files)} files to {OUT}")
    print("By type:", dict(Counter(row[4] for row in rows)))


if __name__ == "__main__":
    main()