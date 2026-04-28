#!/usr/bin/env python3
"""
Build math problem index for similarity-based complexity classification.
Extracts original MATH problems with known levels, embeddings them, and saves FAISS index.
"""

import json
import os
from tqdm import tqdm

# Check for sentence-transformers and faiss
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Installing required packages...")
    os.system("pip install sentence-transformers faiss-cpu -q")
    from sentence_transformers import SentenceTransformer
    import faiss

from src.config import DATA_PATH
from src.data.preprocessing import _normalize_level, classify_complexity

OUTPUT_DIR = DATA_PATH / "math_problem_index"
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.7"))
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def main():
    # TODO - should load from the dataset online, since it should run before load_real_data.py (which generates the openmathinstruct.jsonl file)
    data_path = DATA_PATH / "openmathinstruct.jsonl"

    print(f"Loading data from {data_path}...")

    # Pass 1: Collect original MATH problems with known levels
    math_problems = {}  # problem -> {level, complexity}

    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Finding MATH problems"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            source = str(ex.get("problem_source", "")).lower()

            # Only original MATH (not augmented)
            if source != "math":
                continue

            problem = ex["problem"]
            level = _normalize_level(ex.get("level"))
            if level is None:
                continue # Skip unknown levels
            complexity = classify_complexity(ex)

            # Store unique problems (prefer first occurrence)
            if problem not in math_problems:
                math_problems[problem] = {
                    "level": level,
                    "complexity": complexity,
                }

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