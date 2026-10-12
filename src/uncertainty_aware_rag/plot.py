"""Plot one exam question in 2D (UMAP): where the right drug's chunks are vs. what search returned."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from uncertainty_aware_rag.evaluate import judge, load_questions
from uncertainty_aware_rag.index import get_collection
from uncertainty_aware_rag.expansion import expanded_query

OUT_DIR = Path("docs/figures")
RING_COLORS = {"strict": "green", "loose": "blue", "irrelevant": "black"}


def main(question_id, method="baseline"):
    collection = get_collection()
    data = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(data["embeddings"])
    position = {chunk_id: i for i, chunk_id in enumerate(data["ids"])}

    q = next(q for q in load_questions() if q["id"] == question_id)
    gold_drug = q["id"].rsplit("_q", 1)[0]

    print("Fitting UMAP (first run takes a minute)...")
    reducer = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    points = reducer.embedding_  # 2D position of every chunk

    embed = SentenceTransformerEmbeddingFunction()
    query_2d = reducer.transform(np.array(embed([q["question"]])))
    search_text = q["question"]
    if method == "generated":
        search_text = expanded_query(q["question"])
        expanded_2d = reducer.transform(np.array(embed([search_text])))
    results = collection.query(query_texts=[search_text], n_results=5)

    plt.figure(figsize=(8, 8))

    is_target = np.array([m["drug"] == gold_drug for m in data["metadatas"]])

    plt.scatter(points[~is_target,0], points[~is_target,1], s=5, color="gray", label="other drugs")
    plt.scatter(points[is_target, 0], points[is_target, 1], s=15, color="orange", label=f"{gold_drug} chunks")
    plt.scatter(query_2d[:, 0], query_2d[:, 1], s=200, marker="X", color="red", label="query")
    if method == "generated":
        plt.scatter(expanded_2d[:, 0], expanded_2d[:, 1], s=200, marker="X", color="purple", label="expanded query", zorder=5)
        plt.annotate("", xy=expanded_2d[0], xytext=query_2d[0],
                     arrowprops=dict(arrowstyle="->", color="purple"))

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
    suffix = "" if method == "baseline" else f"_{method}"
    out_path = OUT_DIR / f"{question_id}{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])