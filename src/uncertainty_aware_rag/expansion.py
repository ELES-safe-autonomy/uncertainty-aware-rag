"""Query expansion : search with the question plus an LLM-imagined answer."""

import json
from pathlib import Path

from uncertainty_aware_rag.llm import ask_llm

CACHE = Path("eval/results/generated_answers.json")

ANSWER_INSTRUCTIONS = (
    "You are an expert in FDA drug labeling. Write a short passage (2-4 sentences) "
    "that might appear in an FDA drug label and would answer the question. "
    "Write in the style of a drug label. Output only the passage."
)


def generated_answer(question):
    """Imagined label-style answer. Used ONLY for searching, never shown to anyone."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if question not in cache:
        cache[question] = ask_llm(ANSWER_INSTRUCTIONS, question)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[question]


def expanded_query(question):
    return f"{question} {generated_answer(question)}"