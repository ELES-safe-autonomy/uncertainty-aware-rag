"""The one place this project talks to an LLM. Switch providers by changing only this file."""

import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0, timeout=60)


def ask_llm(instructions, question, max_tries=5):
    """Send instructions + a question to the LLM and return its text reply.
    If the provider is busy or rate-limited (429/503), wait a minute and try again."""
    for attempt in range(1, max_tries + 1):
        try:
            response = _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": question},
                ],
            )
            return response.choices[0].message.content
        except Exception as error:
            if not any(code in str(error) for code in ("429", "503")) or attempt == max_tries:
                raise
            print(f"  LLM busy or rate-limited; waiting 60 s (try {attempt}/{max_tries})...")
            time.sleep(60)