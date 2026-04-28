#!/usr/bin/env python3
import json

from tqdm import tqdm

from src.config import DATA_PATH, DATASET_PATH
from src.data.preprocessing import normalize_problem


print("Loading problem_index.json...")
with open(DATA_PATH / "problem_index.json") as f:
    problem_index = json.load(f)

if isinstance(problem_index, list):
    problem_index_list = problem_index
else:
    problem_index_list = list(problem_index.values())

print(f"Loaded {len(problem_index_list)} problems")

reverse_index = {normalize_problem(item["problem"]): item for item in problem_index_list}

print("Loading DATASET_PATH to get expected_answers...")
problem_to_expected = {}
with open(DATASET_PATH, "r") as f:
    for line in tqdm(f, desc="Loading dataset"):
        ex = json.loads(line)
        problem = normalize_problem(ex.get("problem", ""))
        problem_id = reverse_index[problem]['problem_id']
        if problem_id and problem_id in problem_to_expected and problem_to_expected[problem_id]:
            continue
        problem_to_expected[problem_id] = ex.get("expected_answer", "")

print(f"Loaded {len(problem_to_expected)} expected answers from dataset")

print("Updating problem index with expected_answers...")
for item in tqdm(problem_index_list, desc="Updating"):
    problem = normalize_problem(item.get("problem", ""))
    if problem in reverse_index:
        problem_id = reverse_index[problem]['problem_id']
        expected = problem_to_expected.get(problem_id, "")
        item["expected_answer"] = expected

problem_index_dict = {item["problem_id"]: item for item in problem_index_list}
reverse_index = {item["problem"]: item for item in problem_index_list}

with open(DATA_PATH / "problem_index_dict.json", "w", encoding="utf-8") as f:
    json.dump(problem_index_dict, f, ensure_ascii=False)
print(f"Saved problem index: {len(problem_index_dict)} entries")

with open(DATA_PATH / "problem_index_reverse.json", "w", encoding="utf-8") as f:
    json.dump(reverse_index, f, ensure_ascii=False)
print(f"Saved reverse index")