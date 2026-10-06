"""Check that every evidence_quote in eval/questions.csv appears word-for-word in its label section."""

import csv
import json
import re
from pathlib import Path

QUESTIONS = Path("eval/questions.csv")
LABELS = Path("data/labels")


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    verified, problems = 0, []
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] in ("not_in_label", "out_of_library"):
                continue
            label_file = LABELS / f"{row['id'].rsplit('_q', 1)[0]}.json"
            if not label_file.exists():
                problems.append((row["id"], f"no label file {label_file.name}"))
                continue
            label = json.loads(label_file.read_text(encoding="utf-8"))
            section = label.get(row["label_section"])
            if section is None:
                problems.append((row["id"], f"section '{row['label_section']}' not in label"))
                continue
            if normalize(row["evidence_quote"]) in normalize(" ".join(section)):
                verified += 1
            else:
                problems.append((row["id"], "quote not found word-for-word in that section"))

    print(f"Quotes verified: {verified}")
    print(f"Problems: {len(problems)}")
    for qid, msg in problems:
        print(f"  {qid}: {msg}")


if __name__ == "__main__":
    main()