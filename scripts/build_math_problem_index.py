#!/usr/bin/env python3
"""
Build math problem index for similarity-based complexity classification.
Extracts original MATH problems with known levels from HuggingFace, embeds them, and saves FAISS index.

Can run before load_real_data.py generates openmathinstruct.jsonl.
"""

import json
import os

import faiss
from sentence_transformers import SentenceTransformer

from src.config import DATA_PATH, EMBEDDING_MODEL
from src.data.preprocessing import load_math_problems_with_complexity

OUTPUT_DIR = DATA_PATH / "math_problem_index"
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.6"))


def main():
    print("Loading MATH problems from HuggingFace...")

    # Load MATH problems with level and complexity from HuggingFace
    math_problems = load_math_problems_with_complexity(use_cache=True)

    print(f"Found {len(math_problems):,} unique MATH problems with known levels")

    # Extract problems
    problem_texts = list(math_problems.keys())

    # Pass 2: Load embedding model and encode problems
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(problem_texts):,} problems...")
    embeddings = model.encode(
        problem_texts,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Pass 3: Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings) # type: ignore[attr-defined]

    # Pass 4: Save index and metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {OUTPUT_DIR}...")
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))

    with open(OUTPUT_DIR / "metadata.jsonl", "w") as f:
        for problem in problem_texts:
            f.write(json.dumps({
                "problem": problem,
                "level": math_problems[problem]["level"],
                "complexity": math_problems[problem]["complexity"],
            }, ensure_ascii=False) + "\n")

    # Save config
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "num_problems": len(problem_texts),
        "dimension": dimension,
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done! Index built with {len(problem_texts):,} problems")
    print(f"Index dimension: {dimension}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")


if __name__ == "__main__":
    main()