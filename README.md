# uncertainty-aware-rag

Uncertainty-aware question answering over FDA drug labels.



\# uncertainty-aware-rag



A question-answering assistant over FDA drug labels for critical-care

medications that knows when it isn't sure, and says so instead of guessing.



\## Why

In safety-critical settings, a confident wrong answer is worse than

"I don't know." This project measures whether an AI's confidence can be

trusted, and builds an assistant that acts on it: answer, look further,

or abstain.



\## What it will do

\- \*\*RAG\*\* over 50 FDA drug labels

\- \*\*Agent\*\* with tools: label search, adverse-event counts (openFDA),

&#x20; drug-name normalization (RxNorm)

\- \*\*Confidence + abstention\*\*: every answer comes with a confidence

&#x20; estimate; low confidence leads to "I don't know"

\- \*\*Deployment\*\*: containerized API on a cloud endpoint, with monitoring



\## Status

\- \[x] Repo, environment, secrets handling

\- \[x] 50-label data snapshot (openFDA, 2026-10-04)

\- \[ ] Evaluation questions with answer key

\- \[ ] Phase 1: RAG pipeline + retrieval evaluation

\- \[ ] Phase 2: Agent + tools

\- \[ ] Phase 3: Deployment + monitoring



\## Data

50 IV critical-care drug labels from openFDA (vasopressors, sedatives,

antiarrhythmics, fluids, anticoagulants and more). Selection rules: IV

route only, single-ingredient products, newest label per drug. The drug

list is in `config/drugs.csv`.



\*\*Known limitation:\*\* lactated Ringer's is excluded. Searching openFDA by

ingredient returned D5LR (with dextrose), a different fluid.



\## Quick start

&#x20;   uv sync

&#x20;   copy .env.example .env      # then add your openFDA API key

&#x20;   uv run python scripts/fetch\_labels.py



\## Project structure

&#x20;   config/        drug list (what goes in the library)

&#x20;   data/labels/   downloaded label snapshot

&#x20;   scripts/       data tools (label fetcher)

&#x20;   src/           the system itself (coming in Phase 1)



\*Research prototype. Not medical advice.\*

