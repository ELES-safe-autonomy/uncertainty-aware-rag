"""Load FDA label JSON files into section texts tagged with drug and section."""

import json
from collections import Counter
from pathlib import Path

LABELS_DIR = Path("data/labels")

# Fields that are metadata or packaging, not label text to search.
SKIP_FIELDS = {
    "set_id", "id", "effective_time", "version", "openfda",
    "package_label_principal_display_panel",
    "spl_product_data_elements",
    "spl_unclassified_section",
}

# Subsection -> parent. The parent already contains the subsection's text.
PARENT_OF = {
    "pregnancy": "use_in_specific_populations",
    "pediatric_use": "use_in_specific_populations",
    "geriatric_use": "use_in_specific_populations",
    "nursing_mothers": "use_in_specific_populations",
    "labor_and_delivery": "use_in_specific_populations",
    "mechanism_of_action": "clinical_pharmacology",
    "pharmacodynamics": "clinical_pharmacology",
    "pharmacokinetics": "clinical_pharmacology",
    "carcinogenesis_and_mutagenesis_and_impairment_of_fertility": "nonclinical_toxicology",
    "animal_pharmacology_and_or_toxicology": "nonclinical_toxicology",
}


def is_searchable(field, label):
    if field in SKIP_FIELDS or field.endswith("_table"):
        return False
    parent = PARENT_OF.get(field)
    if parent and parent in label:
        return False  # duplicate: the parent section already has this text
    return True


def load_sections():
    sections = []
    for path in sorted(LABELS_DIR.glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        for field, value in label.items():
            if not isinstance(value, list) or not is_searchable(field, label):
                continue
            text = "\n\n".join(value).strip()
            if text:
                sections.append({"drug": path.stem, "section": field, "text": text})
    return sections


if __name__ == "__main__":
    sections = load_sections()
    drugs = {s["drug"] for s in sections}
    lengths = [len(s["text"]) for s in sections]
    print(f"Sections: {len(sections)} from {len(drugs)} labels")
    print(f"Section length: min {min(lengths)}, max {max(lengths)} characters")
    print(f"Sections longer than 1000 characters: {sum(n > 1000 for n in lengths)}")
    print("Most common sections:", Counter(s["section"] for s in sections).most_common(8))
    print("\nExample:", sections[0]["drug"], "|", sections[0]["section"])
    print(sections[0]["text"][:300])