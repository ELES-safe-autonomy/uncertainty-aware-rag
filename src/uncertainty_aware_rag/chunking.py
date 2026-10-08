"""Split label sections into chunks, keeping drug and section tags on every chunk."""

from collections import Counter

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from uncertainty_aware_rag.ingest import load_sections

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size = 1000,
    chunk_overlap=0
)

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

def split_section(text):
    """Split ONE section's text into a list of chunk strings."""
    chunks = []
    character_split_texts = character_splitter.split_text(text)
    for piece in character_split_texts:
        chunks += token_splitter.split_text(piece)
    return chunks

def make_chunks(sections):
    """Turn tagged sections into tagged chunks."""
    chunks = []
    for s in sections:
        pieces = split_section(s["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{s['drug']}::{s['section']}::{i}",
                "drug": s["drug"],
                "section": s["section"],
                "text": piece,
            })
    return chunks


if __name__ == "__main__":
    chunks = make_chunks(load_sections())
    print(f"Chunks: {len(chunks)}")
    ids = [c["id"] for c in chunks]
    print("All ids unique:", len(ids) == len(set(ids)))
    per_section = Counter((c["drug"], c["section"]) for c in chunks)
    print("Most-split sections:", per_section.most_common(3))
    print("\nExample chunk:", chunks[10]["id"])
    print(chunks[10]["text"][:300])