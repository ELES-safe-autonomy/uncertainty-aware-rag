"""Plot one exam question in 2D (UMAP): where the right drug's chunks are vs. what search returned."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from uncertainty_aware_rag.evaluate import judge, load_questions
from uncertainty_aware_rag.index import get_collection

OUT_DIR = Path("docs/figures")
RING_COLORS = {"strict": "green", "loose": "blue", "irrelevant": "black"}


def main(question_id):
    collection = get_collection()
    data = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(data["embeddings"])
    position = {chunk_id: i for i, chunk_id in enumerate(data["ids"])}

    q = next(q for q in load_questions() if q["id"] == question_id)
    gold_drug = q["id"].rsplit("_q", 1)[0]

    print("Fitting UMAP (first run takes a minute)...")
    reducer = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    points = reducer.embedding_  # 2D position of every chunk

    query_emb = SentenceTransformerEmbeddingFunction()([q["question"]])
    query_2d = reducer.transform(np.array(query_emb))
    results = collection.query(query_texts=[q["question"]], n_results=5)

    plt.figure(figsize=(8, 8))

    # TODO 1: make a True/False array: True where a chunk belongs to gold_drug
    #   hint: np.array([m["drug"] == gold_drug for m in data["metadatas"]])
    is_target = np.array([m["drug"] == gold_drug for m in data["metadatas"]])

    # TODO 2: draw two scatters:
    #   all chunks where NOT is_target -> gray, small (s=5)
    #   all chunks where is_target     -> orange, a bit bigger (s=15)
    #   hint: points[~is_target, 0], points[~is_target, 1]  (~ means "not")
    #   give each a label= for the legend, e.g. label=f"{gold_drug} chunks"
    plt.scatter(points[~is_target,0], points[~is_target,1], s=5, color="gray", label="other drugs")
    plt.scatter(points[is_target, 0], points[is_target, 1], s=15, color="orange", label=f"{gold_drug} chunks")
    plt.scatter(query_2d[:, 0], query_2d[:, 1], s=200, marker="X", color="red", label="query")

    # TODO 3: draw a ring around each of the 5 retrieved chunks
    for rank, (chunk_id, meta) in enumerate(zip(results["ids"][0], results["metadatas"][0]), start=1):
        label = judge(meta, gold_drug, q["label_section"])
        x, y = points[position[chunk_id]]
        plt.scatter(x, y, s=250, facecolors="none", edgecolors=RING_COLORS[label], linewidths=2)
        plt.annotate(str(rank), (x, y), xytext=(8, 8), textcoords="offset points", fontsize=9)

    plt.gca().set_aspect("equal", "datalim")
    plt.axis("off")
    plt.title(q["question"], fontsize=10, wrap=True)
    for name, color in RING_COLORS.items():
        plt.scatter([], [], s=150, facecolors="none", edgecolors=color,
                    linewidths=2, label=f"retrieved: {name}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{question_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])