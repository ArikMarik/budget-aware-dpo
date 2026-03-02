# Implementation Plan: Budget-Aware DPO

**Data Strategy:** Phases 0–6 use **dummy data only** for debugging and sanity checks. Phases 7–11 use **real data** (GSM8K, MATH) for production training, evaluation, and the final report.

# ---

# **Phase 0: Virtual Environment and Cursor Configuration**

**Goal:** Establish an isolated Python environment and configure the AI agent (Cursor) to enforce project-specific rules, formatting standards, and reporting requirements.

1. **Virtual Environment Setup:** Create an isolated virtual environment using Python's venv (e.g., python3 \-m venv .venv). Activate it and upgrade foundational packaging tools (e.g., pip install \--upgrade pip setuptools wheel).  
2. **Cursor Rules Definition:** Create a .cursorrules file in the root directory to control the agent's behavior.  
3. **.cursorrules Content:**  
   Markdown  
   \# Project Constraints & Best Practices (2026)  
   \- **\*\*Environment:\*\*** ALWAYS activate and execute commands within the \`.venv\` virtual environment. Do not install system-wide packages.  
   \- \[cite*\_start\]**\*\*Data Paths:\*\*** Route all data, cache, and model weights to the persistent cluster storage: \`/vol/joberant\_*nobck/data/NLP*\_368307701\_*2526a/\<your\_user\_name\>\`.  
   \- \[cite*\_start\]**\*\*Reproducibility:\*\*** Fix random seeds for all libraries (\`torch\`, \`numpy\`, \`transformers\`) at the start of every script.*  
   *\- **\*\*Reporting:\*\*** After completing a major feature or phase, automatically generate a clear Markdown report (e.g., \`docs/feature\_*reports/report*\_\<feature\>\_*\<date\>.md\`) detailing the implementation, test results, issues resolved, and next steps.  
   \- **\*\*Git Integration:\*\*** Commit changes to Git automatically after each successfully tested phase using clear, descriptive commit messages.  
   \- \[cite*\_start\]**\*\*Documentation:\*\*** Maintain a "white paper" focus in code comments and outputs, omitting exhaustive dead-end work logs\[cite: 210\].*

* **Testing Step:** Ask Cursor to install a harmless package (like cowsay) and verify it installs within the .venv and generates a feature report.  
* **Git Action:** git add .venv/ .cursorrules && git commit \-m "chore: setup virtual environment and cursor rules"

**Senior Engineer Tips:**

* Set your HF\_HOME and PIP\_CACHE\_DIR environment variables to point inside your persistent cluster storage to prevent filling up your home directory quota.

# ---

**Phase 1: Environment Setup and Dummy Data Generation**

**Goal:** Establish the codebase, fix random seeds for reproducibility, and create a minimal dummy dataset to validate the pipeline before executing expensive tasks.

1. **Repository Initialization:** Define a standard directory structure (e.g., data/, src/, notebooks/, scripts/, checkpoints/).  
2. **Library Installation:** Install necessary libraries within the venv, including unsloth or axolotl for training, vllm for inference, transformers, datasets, and peft.

3. **Dummy Data Creation:** Generate a JSONL file with 50 synthetic examples mimicking the OpenMathInstruct-2 format. Include a mix of "Easy" arithmetic questions (similar to GSM8K) and "Hard" mathematical proofs (similar to MATH). Ensure each example has a simulated teacher token count and correctness flag.

4. **Model Loading Check:** Write a script to load Qwen-2.5-0.5B  using Unsloth and run a forward pass on a single dummy example.

* **Checkpointing:** Save the dummy dataset to local storage. Implement an environment variable flag (e.g., USE\_DUMMY\_DATA=1) that routes all subsequent scripts to use this small dataset.  
* **Git Action:** git add . && git commit \-m "chore: setup environment, random seeds, and dummy data generation"

**Senior Engineer Tips:**

* Include a basic shell script (run\_all\_dummy.sh) that will eventually execute the entire pipeline end-to-end on the dummy data in under 2 minutes.

# ---

**Phase 2: Data Preprocessing & 4-Way Augmentation Pipeline**

**Goal:** Classify problems into a complexity matrix and label them for Direct Preference Optimization (DPO).

1. **Complexity Classification:** Implement logic to classify problems into a Complexity Flag ($C$). Problems are labeled as Easy ($C=0$) based on GSM8K origin or low token count, and Hard ($C=1$) based on MATH origin or high token count.

2. **Preference Labeling:**  
   * **Easy-Correct:** Label short, direct paths as Preferred; verbose redundant paths as Rejected.

   * **Hard-Correct:** Label detailed, logically dense CoT paths as Preferred; oversimplified answers as Rejected.

   * **Incorrect Answers:** Label logically flawed paths as Rejected across all complexity levels.

3. **Dataset Statistics:** Compute and log dataset statistics (label distribution, average token lengths for Preferred vs. Rejected per complexity level).

* **Testing Step:** Run the pipeline on the **dummy** data. Manually assert that an Easy problem with a long correct answer is correctly routed to the Rejected column.  
* **Checkpointing:** Serialize the processed dataset to disk. Add logic to skip processing and load from disk if the processed dataset directory already exists.  
* **Git Action:** git add src/data/ && git commit \-m "feat: implement 4-way augmentation and complexity classification pipeline"

**Senior Engineer Tips:**

* When reporting dataset statistics in your final report, include a table showing the token distributions and provide clear examples of samples.

* Ensure that no test set data from GSM8K or MATH leaks into the training pipeline during this preprocessing phase.

# ---

**Phase 3: Custom Loss Function & Sanity Check (Overfitting)**

**Goal:** Implement the Budget-Aware DPO loss function and verify the model can learn it.

1. **Loss Function Modification:** Extend the standard DPO loss to include the Length-Penalty Term ($\\lambda$). Implement the target reward formulation:

   $$R\_{budget}(x, y) \= \\beta \\log \\frac{\\pi\_\\theta(y|x)}{\\pi\_{ref}(y|x)} \- \\lambda(C) \\cdot |y|$$

2. **Dynamic Lambda Routing:** Ensure $\\lambda$ is high when the Complexity Flag $C=0$ (Easy) and near zero when $C=1$ (Hard).

3. **LoRA Configuration:** Configure high-rank LoRA ($r=128+$) for training stability.

4. **Sanity Check Execution:** Train the model on just 10-50 examples from the dummy dataset. Overfit the model to ensure it can successfully memorize the small sample.

* **Testing Step:** Inspect the model outputs from the overfitted checkpoint. Easy prompts must yield highly compressed tokens; Hard prompts must yield full CoT. Show that the sanity experiment worked.

* **Checkpointing:** Save the overfitted LoRA adapter weights.  
* **Git Action:** git add src/models/ && git commit \-m "feat: implement budget-aware DPO loss and verify via overfitting sanity check"

**Senior Engineer Tips:**

* If the model fails to overfit on 50 examples, there is a bug in the custom loss implementation or data loading. Do not proceed to full training until this works.

* Log the length penalty term as a separate metric to ensure it is scaling correctly and not causing exploding gradients.

# ---

**Phase 4: Full Model Training & Baselines**

**Goal:** Train the experimental model and the required baselines for comparison.

1. **Baseline Training:** Train a standard DPO baseline model using the identical dataset but without the length-penalty term. Compare against relevant baselines, such as the most common approach in the literature.

2. **Budget-Aware Training:** Train the primary Qwen-2.5-0.5B model using the custom $R\_{budget}$ loss function.  
3. **Experiment Tracking:** Log all hyperparameters, training loss, and validation loss. Provide access to these exact settings to ensure reproducibility.

* **Testing Step:** Run validation evaluations every N steps during training to catch divergence early.  
* **Checkpointing:** Configure the trainer to save checkpoints periodically, but avoid saving too frequently (e.g., every 500 steps) to prevent blocking the storage space. Implement a resume\_from\_checkpoint argument.

* **Git Action:** git add scripts/training/ && git commit \-m "feat: implement full training loops for budget-aware DPO and baseline models"

**Senior Engineer Tips:**

* Use a naive baseline (like a zero-shot un-fine-tuned model or standard Supervised Fine-Tuning) alongside the standard DPO to effectively frame your results.

# ---

**Phase 5: Evaluation & Benchmarking (Dummy Data — Pipeline Completeness)**

**Goal:** Implement evaluation code and run on dummy data to validate the pipeline. Real evaluation in Phase 9.

1. **Accuracy Evaluation:** Implement logic to evaluate baseline and budget-aware models. On dummy: run on processed test subset.
2. **Efficiency Metrics (TPCA):** Implement Tokens Per Correct Answer calculation.
3. **Latency Benchmarking:** Implement vLLM deployment and time-per-query measurement (optional on dummy).

* **Testing Step:** Run evaluation on dummy checkpoints; verify metric aggregation and decoding work.
* **Checkpointing:** Save raw generations and metric dicts to JSON.
* **Git Action:** git add src/evaluation/ && git commit \-m "feat: implement evaluation for accuracy, TPCA (dummy run)"

**Note:** Phases 0–5 use **dummy data only** for debugging and sanity checks.

# ---

**Phase 6: Visualization (Dummy Data — Pipeline Completeness)**

**Goal:** Implement visualization code and run on dummy outputs. No final report yet (deferred to Phase 11).

1. **Distribution Visualization:** Histograms of response lengths for both models.
2. **Results Table:** Avg. Tokens (Easy/Hard), Reasoning Style comparison.
3. **Figure Export:** PDF format, large fonts.

* **Testing Step:** Generate figures from dummy evaluation outputs.
* **Git Action:** git add src/visualization/ reports/figures_dummy/ && git commit \-m "feat: implement visualization (dummy run)"

**Note:** Final ACL report is **not** part of Phase 6; it is Phase 11 (real data).

# ---

**Phase 7: Real Data Preprocessing**

**Goal:** Integrate real datasets (GSM8K, MATH, OpenMathInstruct-2) and preprocess for training.

1. **Data Loading:** Load GSM8K and MATH from HuggingFace (or equivalent). Load OpenMathInstruct-2 if used.
2. **Format Conversion:** Convert to OpenMathInstruct-2 JSONL format (problem, generated_solution, problem_source, teacher_token_count, correctness_flag).
3. **Train/Test Split:** Ensure no test-set leakage. Hold out GSM8K/MATH test sets for Phase 9 evaluation.
4. **Preprocessing:** Run existing 4-way augmentation pipeline on real data. Output to `data/processed_dpo_dataset_real/`.
5. **Config:** Add `USE_DUMMY_DATA=0` path routing; `get_input_path()` returns real dataset when not dummy.

* **Checkpointing:** Save processed real dataset to cluster storage.
* **Git Action:** git add src/data/ scripts/load_real_data.py && git commit \-m "feat: integrate real data (GSM8K, MATH) preprocessing"

# ---

**Phase 8: Full Training on Real Data**

**Goal:** Repeat Phase 4 training using real processed data.

1. **Baseline DPO:** Train on real DPO pairs (no length penalty).
2. **Budget-Aware DPO:** Train on real DPO pairs with R_budget loss.
3. **Experiment Tracking:** Log hyperparameters, loss, validation. Reproducible configs.

* **Checkpointing:** Save to `checkpoints/baseline_dpo_real/`, `checkpoints/budget_aware_dpo_real/`.
* **Git Action:** git add scripts/training/ && git commit \-m "feat: full training on real data"

# ---

**Phase 9: Evaluation on Real Data**

**Goal:** Repeat Phase 5 evaluation on GSM8K and MATH test sets using real-data checkpoints.

1. **Accuracy:** Evaluate on GSM8K and MATH test sets. Track MATH Level 4–5 retention.
2. **TPCA:** Tokens Per Correct Answer for all models.
3. **Latency:** vLLM deployment, time-per-query.

* **Checkpointing:** Save evaluation outputs to JSON.
* **Git Action:** git add src/evaluation/ && git commit \-m "feat: evaluation on real data (GSM8K, MATH)"

# ---

**Phase 10: Visualization on Real Data**

**Goal:** Generate publication-ready figures from real evaluation results.

1. **Distribution Visualization:** Histograms from real model outputs.
2. **Results Table:** Final comparison table.
3. **Figure Export:** PDF, ACL-ready.

* **Git Action:** git add reports/figures/ && git commit \-m "feat: visualization from real data"

# ---

**Phase 11: Final Report**

**Goal:** Compile the final ACL-formatted project report using real-data results.

1. **Report Drafting:** ACL template, ≤8 pages. Introduction, Background, Method, Experimental Design, Results, Discussion.
2. **Formatting:** PDF figures, algorithm box, bibliography.
3. **Self-Contained:** Code link, no critical content in appendix.

* **Git Action:** git add reports/ && git commit \-m "docs: final ACL report (real data)"

---

# Implementation Progress

**Last Updated:** 2026-03-02

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 | ✅ Complete | Virtual env, .cursorrules (dummy) |
| Phase 1 | ✅ Complete | Dummy data (50 examples), run_all_dummy.sh |
| Phase 2 | ✅ Complete | 4-way augmentation on **dummy** data |
| Phase 3 | ✅ Complete | Budget-aware DPO, sanity check on **dummy** |
| Phase 4 | ✅ Complete | Training scripts, run on **dummy** |
| Phase 5 | ✅ Complete | Evaluation code, run on **dummy** |
| Phase 6 | ✅ Complete | Visualization code, run on **dummy** (no final report) |
| Phase 7 | ✅ Complete | Real data (OpenMathInstruct-2), synthetic DPO pairs |
| Phase 8 | ⏳ Pending | Full training on **real** data |
| Phase 9 | ⏳ Pending | Evaluation on **real** data |
| Phase 10 | ⏳ Pending | Visualization from **real** data |
| Phase 11 | ⏳ Pending | Final ACL report (**real** data) |

---

# User notes and todos for the future

- [ ] **Dataset imbalance:** Phase 7 real data preprocessing produced ~85% Hard pairs (742 easy, 4252 hard). Investigate whether this imbalance affects training quality/evaluation and consider rebalancing (e.g., oversampling easy, stratified sampling, or loading more GSM8K-heavy subsets).
