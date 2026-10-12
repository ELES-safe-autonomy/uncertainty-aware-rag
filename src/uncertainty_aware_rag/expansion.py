"""Query expansion : search with the question plus an LLM-imagined answer."""

import re
import json
from pathlib import Path

from uncertainty_aware_rag.llm import ask_llm

CACHE = Path("eval/results/generated_answers.json")

ANSWER_INSTRUCTIONS = (
    "You help a search engine find passages in FDA drug labels. "
    "Given a clinician's question, write a short passage (2-4 sentences) in the style "
    "of the FDA drug label section that would contain the answer. "
    "This text is used only to search a database of real drug labels; it is never "
    "shown to patients or clinicians and is not medical advice. "
    "Use typical label wording and section names (for example Dosage and Administration, "
    "Warnings and Precautions, Use in Specific Populations). Output only the passage."
)


def generated_answer(question):
    """Imagined label-style answer. Used ONLY for searching, never shown to anyone."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if question not in cache:
        cache[question] = ask_llm(ANSWER_INSTRUCTIONS, question)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[question]


REFUSAL_PHRASES = ("sorry", "can't", "can\u2019t", "cannot provide", "unable to")


def is_refusal(text):
    """True if the LLM declined instead of writing a label-style passage."""
    return any(phrase in text.lower() for phrase in REFUSAL_PHRASES)


def expanded_query(question):
    """Question + imagined answer; falls back to the plain question if the LLM refused."""
    answer = generated_answer(question)
    if is_refusal(answer):
        return question
    return f"{question} {answer}"


QUERIES_CACHE = Path("eval/results/generated_queries.json")

QUERIES_INSTRUCTIONS = (
    "You help a search engine find passages in FDA drug labels. "
    "Given a clinician's question, write up to 5 short related search questions that "
    "cover different ways the label might word the answer. "
    "Keep the drug name in every question. "
    "These are used only to search a database of real drug labels; they are never "
    "shown to patients or clinicians and are not medical advice. "
    "Output one question per line, with no numbering and nothing else."
)


def related_queries(question):
    """Up to 5 LLM-written rewrites of the question (cached)."""
    cache = json.loads(QUERIES_CACHE.read_text(encoding="utf-8")) if QUERIES_CACHE.exists() else {}
    if question not in cache:
        text = ask_llm(QUERIES_INSTRUCTIONS, question)
        lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines()]
        cache[question] = [line for line in lines if line]
        QUERIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        QUERIES_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return [q for q in cache[question] if not is_refusal(q)][:5]


def multi_query_search(collection, question, k=5, pool=10):
    """Search with the question + its rewrites; merge with Reciprocal Rank Fusion."""
    queries = [question] + related_queries(question)
    results = collection.query(query_texts=queries, n_results=pool)
    scores, metas = {}, {}
    for ids, ms in zip(results["ids"], results["metadatas"]):
        for rank, (chunk_id, meta) in enumerate(zip(ids, ms), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
            metas[chunk_id] = meta
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [metas[c] for c in best]