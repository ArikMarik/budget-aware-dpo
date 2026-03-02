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

* **Testing Step:** Run the pipeline on the dummy data. Manually assert that an Easy problem with a long correct answer is correctly routed to the Rejected column.  
* **Checkpointing:** Serialize the processed HuggingFace dataset to disk. Add logic to skip processing and load from disk if the processed dataset directory already exists.  
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

**Phase 5: Evaluation & Benchmarking**

**Goal:** Measure the Pareto frontier between mathematical accuracy and computational efficiency.

1. **Accuracy Evaluation:** Evaluate both the baseline and experimental models on the GSM8K and MATH test sets. Track the expected retention of \>95% accuracy on MATH Level 4-5.

2. **Efficiency Metrics (TPCA):** Calculate the Tokens Per Correct Answer (TPCA) for all models. Verify the hypothesized 30-50% reduction in TPCA on GSM8K.

3. **Latency Benchmarking:** Deploy the fine-tuned models using vLLM to measure real-world inference latency (time-per-query).

* **Testing Step:** Run the evaluation script on a 5-shot subset of the test sets to ensure metric aggregations and generation decoding schemes do not throw errors.  
* **Checkpointing:** Save raw model generations and computed metric dictionaries to JSON files before passing them to visualization scripts.  
* **Git Action:** git add src/evaluation/ && git commit \-m "feat: implement evaluation for accuracy, TPCA, and vLLM latency"

**Senior Engineer Tips:**

* Always account for randomness during evaluation. Record the seed used and note the specific decoding scheme (e.g., greedy decoding vs. temperature sampling).

* If you encounter negative results (e.g., accuracy drops significantly), ensure they are thoroughly analyzed in the final report, as well-analyzed negative results are valued if the methodology is sound.

# ---

**Phase 6: Visualization and Final Report**

**Goal:** Generate publication-ready figures and draft the final project report in the required ACL format.

1. **Distribution Visualization:** Create histograms showing the response lengths for both models. Plot the expected bimodal distribution demonstrating "Learned Routing" (20-40 tokens for simple, 300+ tokens for complex).

2. **Results Formatting:** Construct a table comparing Avg. Tokens (Easy), Avg. Tokens (Hard), and Reasoning Style between the baseline and the Budget-Aware DPO model. Ensure results are given in an easy-to-understand format.

3. **Report Drafting:** Compile the report using the ACL template, remaining strictly under the 8-page limit. Include Introduction, Background, Method, Experimental Design, Results, and Discussion.

4. **Formatting Constraints:** Ensure all figures are added as PDF files (not JPEG/PNG) and use large enough fonts. Include an algorithm box outlining the method on the first or second page. Include full bibliography citations.

* **Testing Step:** Compile the document and strictly verify it does not exceed 8 pages (excluding bibliography and AI disclosure).

* **Git Action:** git add reports/ && git commit \-m "docs: finalize data visualizations and compile ACL formatted report"

**Senior Engineer Tips:**

* Write the report as a white paper. Avoid exhaustive work logs (e.g., "We tried X but it failed..."); be concise and present the final effective methodology.

* The appendix should not contain content critical to the flow of the paper, as it is assumed it will not be read unless specifically sought out.

* Host your code on GitHub and ensure a link is provided in the PDF. Ensure the paper is entirely self-contained.

---

# Implementation Progress

**Last Updated:** 2026-03-02

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 | ✅ Complete | Virtual env, .cursorrules, cowsay test, feature report |
| Phase 1 | ✅ Complete | Directory structure, requirements, dummy data (50 examples), model load check script, run_all_dummy.sh |
| Phase 2 | ✅ Complete | 4-way augmentation, complexity classification, DPO pairs, dataset stats, checkpointing |
| Phase 3 | 🔄 In Progress | Budget-aware DPO loss implemented; LoRA config in sanity script. **Sanity check (overfitting) not yet run** — will restart on GPU. |
| Phase 4 | ⏳ Pending | Full training loops |
| Phase 5 | ⏳ Pending | Evaluation & benchmarking |
| Phase 6 | ⏳ Pending | Visualization & final report |

**Current Blocker:** Sanity check was accidentally run on CPU; needs to be restarted on GPU.