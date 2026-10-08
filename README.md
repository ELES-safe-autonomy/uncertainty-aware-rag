# uncertainty-aware-rag

A question-answering assistant over FDA drug labels for critical-care
medications that knows when it isn't sure, and says so instead of guessing.

*Research prototype. Not medical advice.*

## Why
In safety-critical settings, a confident wrong answer is worse than
"I don't know." This project builds a RAG system over drug labels,
measures where it fails, and adds an uncertainty layer so it can answer,
look further, or abstain.

## Status
- [x] Repo, environment (uv), secrets handling
- [x] Data: 47 openFDA drug labels, audited and frozen as a snapshot
- [x] Evaluation set: 44 questions with an answer key
- [x] RAG baseline: loading, section-aware chunking, Chroma index, first query
- [ ] Retrieval evaluation on the full question set
- [ ] Retrieval improvements (filtering, query expansion, re-ranking)
- [ ] Answer generation with citations and abstention
- [ ] Agent with tools (label search, adverse events, drug-name normalization)
- [ ] Deployment and monitoring

## Data
47 IV critical-care drug labels from openFDA: vasopressors, inotropes,
sedatives, analgesics, paralytics, antiarrhythmics, antihypertensives,
fluids, anticoagulants and electrolytes. The drug list is in
`config/drugs.csv`.

**Selection rules:** IV route only, single-ingredient products, newest
label per drug.

**Manual audit:** automated rules let through wrong products with
matching generic names. Fixed or excluded:
- albumin matched a radioactive imaging agent (fixed: search "albumin human")
- sodium chloride matched a dexmedetomidine premix (excluded)
- lidocaine matched a local anesthetic, not the IV antiarrhythmic (excluded)
- alteplase matched Cathflo, a catheter-clearing product (excluded)
- lactated Ringer's matched D5LR, a different fluid (excluded)

## Evaluation set
44 questions in `eval/questions.csv`:

| Type | Count | Purpose |
|---|---|---|
| Answerable | 24 | The label clearly states the answer |
| Ambiguous | 8 | The label is hedged or incomplete |
| Not in label | 8 | Realistic question the label doesn't answer |
| Out of library | 4 | Drug not in the library at all; should abstain |

Questions were drafted by an LLM from 8 focus-drug labels, with every
evidence quote verified word for word against the source
(`scripts/check_quotes.py`), then reviewed by a domain expert.
Out-of-library questions were written by hand.

## RAG pipeline (baseline)
    labels (JSON) -> sections -> chunks -> embeddings -> Chroma -> top-5 search

1. **Load** (`ingest.py`): 729 sections from 47 labels. Metadata and
   tables are skipped; subsections already contained in their parent
   section are dropped to avoid duplicates.
2. **Chunk** (`chunking.py`): each section is split on its own (never
   across sections or labels), 1000 characters then 256 tokens, giving
   2516 chunks tagged with drug and section.
3. **Index and search** (`index.py`): Chroma with the
   all-MiniLM-L6-v2 embedding model.

## Findings so far
**1. Retrieval finds the topic, not the drug.** Query: "What is the
maximum infusion rate for norepinephrine?" Only 2 of the top 5 results
came from norepinephrine, and neither was its dosage section. The other
3 were dosage sections of different drugs (nitroprusside, heparin,
procainamide). Embeddings capture *what kind of question* more strongly
than *which drug*. Planned fixes: filter by drug, hybrid keyword search,
re-ranking.

**2. The token splitter alters text.** The splitter lowercases text and
changes spacing. In one chunk, "1,500 units/hour" became
"1, 500 units / hour". In a dosing domain this is unacceptable. Planned
fix: a splitter that returns exact slices of the original text, plus a
test that every chunk appears word for word in its source.

## Quick start
    uv sync
    copy .env.example .env      # then add your openFDA API key
    uv run python scripts/fetch_labels.py
    uv run python -m uncertainty_aware_rag.index

## Project structure
    config/        drug list
    data/labels/   label snapshot (openFDA, Oct 2026)
    eval/          evaluation questions
    scripts/       data tools: fetcher, question merge, quote check
    src/           the pipeline: ingest, chunking, index
