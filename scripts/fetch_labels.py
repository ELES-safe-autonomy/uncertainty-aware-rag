"""Download one FDA drug label per drug in config/drugs.csv and save it to data/labels/."""

import csv
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENFDA_API_KEY")
LABEL_URL = "https://api.fda.gov/drug/label.json"
DRUG_LIST = Path("config/drugs.csv")
OUT_DIR = Path("data/labels")


def load_drugs():
    with DRUG_LIST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search_labels(name):
    query = f'openfda.generic_name:"{name}" AND openfda.route:"INTRAVENOUS"'
    params = {"search": query, "limit": 100, "api_key": API_KEY}
    response = requests.get(LABEL_URL, params=params, timeout=30)
    if response.status_code == 404:  # openFDA uses 404 to mean "no matches"
        return []
    if not response.ok:  # our own message: never show the URL (it contains the key)
        raise RuntimeError(f"openFDA error {response.status_code} for {name!r}")
    return response.json()["results"]


def pick_label(results):
    single = [r for r in results
              if len(r.get("openfda", {}).get("substance_name", [])) == 1]
    if not single:
        return None
    return max(single, key=lambda r: r.get("effective_time", ""))


def main():
    drugs = load_drugs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, not_found, only_combos = [], [], []

    for drug in drugs:
        name = drug["generic_name"].strip()
        query_name = (drug.get("search_name") or name).strip()
        results = search_labels(query_name)
        if not results:
            not_found.append(name)
            continue
        label = pick_label(results)
        if label is None:
            only_combos.append(name)
            continue
        out_path = OUT_DIR / f"{name.replace(' ', '_')}.json"
        out_path.write_text(json.dumps(label, indent=2), encoding="utf-8")
        saved.append(name)

    print(f"\nSaved: {len(saved)} of {len(drugs)}")
    print(f"Not found ({len(not_found)}): {not_found}")
    print(f"Only combination products ({len(only_combos)}): {only_combos}")


if __name__ == "__main__":
    main()