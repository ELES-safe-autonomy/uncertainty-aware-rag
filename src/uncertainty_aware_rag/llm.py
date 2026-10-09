"""The one place this project talks to an LLM. Switch providers by changing only this file."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
_client = genai.Client()  # reads GEMINI_API_KEY from the environment


def ask_llm(instructions, question):
    """Send instructions + a question to the LLM and return its text reply."""
    interaction = _client.interactions.create(
        model=MODEL,
        input=f"{instructions}\n\n{question}",
    )
    return interaction.output_text