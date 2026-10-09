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
- [x] RAG baseline: loading, section-aware chunking, Chroma index
- [x] Retrieval evaluation on the full question set
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
3. **Index and search** (`index.py`): Chroma with the all-MiniLM-L6-v2
   embedding model.

## Retrieval baseline (32 questions)
Every answerable and ambiguous question was run through search (top 5).
Each retrieved chunk is judged automatically against the answer key:

- **strict**: right drug **and** right label section
- **loose**: right drug, any section
- **irrelevant**: wrong drug

| Metric | Strict | Loose | Plain meaning |
|---|---|---|---|
| Hit rate@5 | 0.66 | 0.91 | Search finds the right **drug** for 91% of questions, but the right **section** for only 66%. |
| Precision@5 | 0.20 | 0.60 | On average, only 1 of the 5 results is the right section; 3 of 5 are the right drug. |
| MRR | 0.46 | 0.78 | The first right-section result typically appears around rank 2. |

**Main finding:** search usually finds the right drug, then often picks
the wrong part of its label. For about a third of questions, the exact
answer location never appears in the top 5. Per-question results are in
`eval/results/retrieval_baseline.csv`.

### What retrieval looks like
Each chunk's embedding is projected to 2D with UMAP. Gray: all chunks.
Orange: chunks of the drug the question is about. Red X: the question.
Rings: the 5 retrieved chunks (green = strict, blue = loose,
black = irrelevant), numbered by rank.

**Success (norepinephrine starting rate).** The question lands among
norepinephrine's dosing chunks; ranks 1 and 3 are the right section.
Ranks 2, 4 and 5 are other drugs.
![success](docs/figures/NOREPINEPHRINE_ID.png)

**Right drug, wrong section (propofol fat and calories).** The question
lands inside propofol's dense cluster and every result is propofol, but
mostly from the wrong part of the label.
![wrong section](docs/figures/PROPOFOL_ID.png)

**Wrong drug (vasopressin in pregnancy).** The question lands in a
cluster of *other drugs' pregnancy sections*: ranks 1 to 4 are other
drugs. Only rank 5 is vasopressin, and from the wrong section. The word
"pregnant" pulled the search more strongly than the drug name.
![wrong drug](docs/figures/VASOPRESSIN_ID.png)

The orange dots are spread across the whole map: the embedding groups
text by **type of section** (dosing near dosing, pregnancy near
pregnancy) much more than by **drug**. That is the root cause of
wrong-drug retrieval.

*Caution:* UMAP squeezes 384 dimensions into 2, so distances in these
plots are distorted. The figures are for intuition; the metrics table
is the evidence.

## Known issues
- **The token splitter alters text.** It lowercases text and changes
  spacing: in one chunk, "1,500 units/hour" became "1, 500 units / hour".
  Planned fix: a splitter that returns exact slices of the original text,
  plus a test that every chunk appears word for word in its source.

## Next steps
- Retrieval improvements, each measured against this baseline:
  filtering by drug, query expansion, re-ranking.
- Fix the token splitter and re-measure.

## Quick start
    uv sync
    copy .env.example .env      # then add your openFDA API key
    uv run python scripts/fetch_labels.py
    uv run python -m uncertainty_aware_rag.index
    uv run python -m uncertainty_aware_rag.evaluate
    uv run python -m uncertainty_aware_rag.plot <question_id>

## Project structure
    config/         drug list
    data/labels/    label snapshot (openFDA, Oct 2026)
    eval/           evaluation questions and results
    docs/figures/   retrieval plots
    scripts/        data tools: fetcher, question merge, quote check
    src/            the pipeline: ingest, chunking, index, evaluate, plot
