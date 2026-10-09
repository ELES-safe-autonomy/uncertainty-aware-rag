"""Evaluate retrieval on the exam: precision@5, hit rate@5 and MRR (strict and loose)."""

import csv
from pathlib import Path

from uncertainty_aware_rag.index import get_collection, search
from uncertainty_aware_rag.ingest import PARENT_OF

QUESTIONS = Path("eval/questions.csv")
RESULTS_CSV = Path("eval/results/retrieval_baseline.csv")
K = 5


def family(section):
    """Map a subsection to its parent: 'pharmacokinetics' -> 'clinical_pharmacology'."""
    return PARENT_OF.get(section, section)


def load_questions():
    """Only questions whose answer location is known (answerable + ambiguous)."""
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["label_section"]]


def judge(meta, gold_drug, gold_section):
    """Label one retrieved chunk as 'strict', 'loose' or 'irrelevant'."""
    right_drug = meta["drug"] == gold_drug
    right_section = family(meta["section"]) == family(gold_section)

    if right_drug and right_section:
        return "strict"
    if right_drug:
        return "loose"
    return "irrelevant"


def reciprocal_rank(labels, wanted):
    """1/rank of the first label that is in `wanted`; 0 if none is."""
    # TODO 2: loop with `for rank, label in enumerate(labels, start=1):`
    #         return 1 / rank at the first label that is in `wanted`
    #         after the loop, return 0
    ...
    for rank, label in enumerate(labels, start=1):
        if label in wanted: 
            return 1/rank
    return 0

def main():
    collection = get_collection()
    rows = []
    for q in load_questions():
        gold_drug = q["id"].rsplit("_q", 1)[0]
        results = search(collection, q["question"], n_results=K)
        labels = [judge(m, gold_drug, q["label_section"]) for m in results["metadatas"][0]]
        n_strict = labels.count("strict")
        n_loose = n_strict + labels.count("loose")  # loose includes strict
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "strict_at_5": n_strict,
            "loose_at_5": n_loose,
            "hit_strict": int(n_strict > 0),
            "hit_loose": int(n_loose > 0),
            "rr_strict": reciprocal_rank(labels, {"strict"}),
            "rr_loose": reciprocal_rank(labels, {"strict", "loose"}),
        })

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)

    def mean(key):
        return sum(r[key] for r in rows) / n

    print(f"Questions evaluated: {n}\n")
    print(f"{'metric':<14}{'strict':>8}{'loose':>8}")
    print(f"{'precision@5':<14}{mean('strict_at_5') / K:>8.2f}{mean('loose_at_5') / K:>8.2f}")
    print(f"{'hit rate@5':<14}{mean('hit_strict'):>8.2f}{mean('hit_loose'):>8.2f}")
    print(f"{'MRR':<14}{mean('rr_strict'):>8.2f}{mean('rr_loose'):>8.2f}")
    print(f"\nPer-question results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()