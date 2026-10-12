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
- [x] Retrieval improvement: query expansion (imaginary answer, multiple queries)
- [ ] Retrieval improvement: re-ranking with a cross-encoder
- [ ] Retrieval improvement: filtering by drug
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

<table>
  <tr>
    <td align="center"><b>Success</b><br>norepinephrine starting rate</td>
    <td align="center"><b>Right drug, wrong section</b><br>propofol fat and calories</td>
    <td align="center"><b>Wrong drug</b><br>vasopressin in pregnancy</td>
  </tr>
  <tr>
    <td><img src="docs/figures/norepinephrine_q1.png" width="300"></td>
    <td><img src="docs/figures/propofol_q3.png" width="300"></td>
    <td><img src="docs/figures/vasopressin_q4.png" width="300"></td>
  </tr>
</table>

- **Success:** the question lands among norepinephrine's dosing chunks;
  ranks 1 and 3 are the right section, ranks 2, 4 and 5 are other drugs.
- **Right drug, wrong section:** every result is propofol, but mostly from
  the wrong part of the label.
- **Wrong drug:** the question lands among *other drugs' pregnancy
  sections*; ranks 1 to 4 are other drugs. Only rank 5 is vasopressin,
  from the wrong section. "Pregnant" pulled the search more strongly than
  the drug name.

*Click any figure to see it full size.*

The orange dots are spread across the whole map: the embedding groups
text by **type of section** (dosing near dosing, pregnancy near
pregnancy) much more than by **drug**. That is the root cause of
wrong-drug retrieval.

*Caution:* UMAP squeezes 384 dimensions into 2, so distances in these
plots are distorted. The figures are for intuition; the metrics table
is the evidence.

## Retrieval improvements: query expansion
A clinician's question and the label text that answers it are written in
different styles (short and conversational vs. formal label language).
The embedding compares style as well as meaning, so the question often
lands near the right drug but not on the right section. Query expansion
uses an LLM to rewrite the query before searching. Two methods were
tried, as taught in the DeepLearning.AI course *Advanced Retrieval for AI
with Chroma*.

**Method 1: imaginary answer** (also called HyDE). The LLM writes a short
passage that *sounds like* the label section that would answer the
question; the search uses the question plus this passage. The passage
does not need to be correct, only to sound like the right section. It is
used for search only and is never shown to anyone.

**Method 2: multiple queries.** The LLM writes up to 5 related questions.
The original and each rewrite are searched separately, and the 6 result
lists are merged with Reciprocal Rank Fusion (RRF): a chunk ranks higher
if it appears near the top of several lists.

Both use `openai/gpt-oss-120b` via Groq (`llm.py`). Every LLM output is
cached in `eval/results/` so runs are reproducible, and calls are retried
only on temporary errors (429, 503).

### Results (32 questions)

| Strict / loose | Baseline | Imaginary answer | Multiple queries |
|---|---|---|---|
| Hit rate@5 | 0.66 / 0.91 | **0.78 / 0.94** | 0.69 / 0.88 |
| Precision@5 | 0.20 / 0.60 | **0.28 / 0.69** | 0.23 / 0.62 |
| MRR | 0.46 / 0.78 | **0.62 / 0.83** | 0.48 / 0.78 |

**Imaginary answer wins on every metric:** 4 questions gained the right
section in the top 5 and none lost it. Writing in label language moves
the query into the right part of the map:

<table>
  <tr>
    <td align="center"><b>Baseline</b><br>question only</td>
    <td align="center"><b>Imaginary answer</b><br>question + generated passage (purple X)</td>
  </tr>
  <tr>
    <td><img src="docs/figures/propofol_q1.png" width="400"></td>
    <td><img src="docs/figures/propofol_q1_generated.png" width="400"></td>
  </tr>
</table>

Baseline: the question lands just outside the propofol cluster; no result
is the right section and 2 of 5 are other drugs. Imaginary answer: the
query moves into the cluster; all 5 results are propofol and the right
section is ranked first.

**Multiple queries barely beats the baseline:** 4 gains (one, nicardipine_q4,
fixed by no other method) but 3 losses. More queries bring in more
candidates, but also more noise, and simple voting does not pick the best
5. This method is meant to be paired with re-ranking (next step), which
reorders a wide pool of candidates.

### Failures found along the way
- **Refusals.** The first imaginary-answer instructions made the LLM
  refuse 3 of 32 questions as "medical advice," and the refusal text
  ("I'm sorry, I can't...") was searched as if it were label text.
  Fixed by a fallback to the plain question, then by explaining the real
  task (search text, never shown to anyone): 0 refusals.
- **Invented drugs.** A first version of the multi-query instructions
  said "keep the drug name in every question." When the question named
  no drug, the LLM made one up: all 5 rewrites of a vasopressin question
  said "norepinephrine," and RRF treated the repeated mistake as
  agreement. "Dex" was also read as dexamethasone instead of
  dexmedetomidine. Removing that rule fixed the invented names, but
  traded some strict accuracy for wider coverage.
- **Questions without a drug name.** Some evaluation questions ("Can I
  run *this* through the same line...") only make sense if the drug is
  known from context. In a real app the drug would come from the screen
  the clinician is on; this motivates filtering by drug.

**Lesson:** an LLM inside a pipeline can fail silently. Its output must be
checked before it is used, and a method that repeats the same LLM call
can amplify one mistake.

### Limits
- 32 questions: one question moves hit rate by 3 points. The instructions
  were not tuned further, to avoid fitting them to this test set.
- LLM output varies between runs; caching fixes one run for comparison.
- Imaginary answers can contain made-up doses. Acceptable only because
  they are used for search and never shown.
- One extra LLM call per question (cost and latency).

## Known issues
- **The token splitter alters text.** It lowercases text and changes
  spacing: in one chunk, "1,500 units/hour" became "1, 500 units / hour".
  Planned fix: a splitter that returns exact slices of the original text,
  plus a test that every chunk appears word for word in its source.

## Next steps
- Re-ranking with a cross-encoder, replacing RRF voting for multiple
  queries.
- Filtering by drug, then hybrid search.
- Fix the token splitter and re-measure.

## Quick start
    uv sync
    copy .env.example .env      # then add your openFDA and Groq API keys
    uv run python scripts/fetch_labels.py
    uv run python -m uncertainty_aware_rag.index
    uv run python -m uncertainty_aware_rag.evaluate baseline
    uv run python -m uncertainty_aware_rag.evaluate generated
    uv run python -m uncertainty_aware_rag.evaluate multi
    uv run python -m uncertainty_aware_rag.plot propofol_q1
    uv run python -m uncertainty_aware_rag.plot propofol_q1 generated

## Project structure
    config/         drug list
    data/labels/    label snapshot (openFDA, Oct 2026)
    eval/           evaluation questions and results (incl. cached LLM answers)
    docs/figures/   retrieval plots
    scripts/        data tools: fetcher, question merge, quote check
    src/            the pipeline: ingest, chunking, index, llm, expansion,
                    evaluate, plot
