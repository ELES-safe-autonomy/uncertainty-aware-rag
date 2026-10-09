"""Build a Chroma collection from tagged chunks and run a test query."""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from uncertainty_aware_rag.chunking import make_chunks
from uncertainty_aware_rag.ingest import load_sections

DB_PATH = "chroma_db"
COLLECTION_NAME = "drug_labels"


def build_collection():
    chunks = make_chunks(load_sections())
    embedding_function = SentenceTransformerEmbeddingFunction()
    client = chromadb.PersistentClient(path=DB_PATH)

    # TODO 1: start fresh, so re-running doesn't add the same chunks twice.
    #   a) delete the old collection if it exists
    #      (hint: client.delete_collection(COLLECTION_NAME) inside try/except)
    #   b) create a new one with the embedding function (hint: the lab's create_collection)
    try: 
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_function)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"drug":c["drug"],"section":c["section"]} for c in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection


def search(collection, query, n_results=5):
    return collection.query(query_texts=[query], n_results=n_results)

def get_collection():
    """Open the existing collection without rebuilding it."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(
        COLLECTION_NAME, embedding_function=SentenceTransformerEmbeddingFunction()
    )

if __name__ == "__main__":
    collection = build_collection()
    print("Chunks in collection:", collection.count())

    query = "What is the maximum infusion rate for norepinephrine?"
    results = search(collection, query)

    for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
        print(f"{rank}. [{dist:.3f}] {meta['drug']} | {meta['section']}")
        print(f"   {doc[:200]}\n")