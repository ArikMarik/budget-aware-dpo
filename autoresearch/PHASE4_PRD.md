# Phase 4 PRD — Baseline Establishment & SFT Accuracy Push

**Date**: 2026-03-29
**Branch**: `autoresearch/mar26`
**GPU**: 1 GPU available

---

## 1. Motivation

After 10 iterations of DPO training, our best model gets 29.2% on easy (GSM8K-like) problems. The published benchmark for Qwen2.5-0.5B base is ~44% on GSM8K. We may be **degrading** the model with DPO. Before any further budget-aware work, we need:

1. **Ground truth**: What does the base model actually score on our eval pipeline?
2. **SFT ceiling**: How high can supervised fine-tuning push accuracy?

Token count is **not a goal** in this phase — only accuracy. Token statistics should be documented but not optimized for.

---

## 2. Goals

| Priority | Goal | Success Metric |
|----------|------|----------------|
| P0 | Evaluate base Qwen2.5-0.5B (no training) on our full eval pipeline | Get a number with confidence |
| P0 | Understand any gap vs published 44% GSM8K | Diagnose root cause if gap exists |
| P1 | SFT fine-tune on our data, maximize accuracy | Beat base model accuracy |
| P1 | Find optimal SFT hyperparameters via autoresearch loop | Documented sweep |

---

## 3. Task A — Base Model Evaluation

### 3.1 What to Build

A script `scripts/eval_base_model.py` that:
- Loads `Qwen/Qwen2.5-0.5B` directly (no LoRA, no checkpoint)
- Runs the full tiered evaluation (Tier 0 + Tier 1 + Tier 2 LLM judge)
- Uses `--use-real` mode (GSM8K test + MATH test)
- Evaluates on **all available problems** (1,319 GSM8K + 5,000 MATH = 6,319 total), not a 500 sample — we want tight confidence intervals
- Reports: overall accuracy, easy accuracy (GSM8K), hard accuracy (MATH), MATH by level, avg tokens easy/hard, TPCA
- Saves full results JSON

### 3.2 Why Full Dataset

Previous evals used 500 samples (250 easy + 250 hard). With 250 GSM8K samples, a 44% accuracy has a 95% CI of ~38-50%. We need the full 1,319 GSM8K problems for a reliable number. The 5,000 MATH problems give us per-level breakdown.

### 3.3 Implementation Notes

- The existing `evaluate_checkpoint()` in `run_evaluation.py` loads a LoRA checkpoint on top of a base model. The new script should load the base model **directly** and call `generate_and_evaluate()` without `PeftModel`.
- Keep max_new_tokens=256 first (matches our previous evals). Then also run with max_new_tokens=512 to check if truncation hurts accuracy.
- Use same prompt format: `"Problem: {problem}\nSolution:"`

### 3.4 Potential Gap Sources

If we don't match 44%, investigate:
- **Prompt format**: Published benchmarks may use few-shot prompting (e.g., 8-shot GSM8K). Our eval uses 0-shot.
- **max_new_tokens**: 256 may truncate solutions. GSM8K solutions average ~150 tokens but some are longer.
- **Answer extraction**: Our extraction pipeline may miss correct answers in unfamiliar formats.
- **do_sample=False**: Should match published (greedy decoding is standard).
- **LoRA overhead**: Run the base model with and without an untrained LoRA adapter to check if PeftModel itself degrades performance. If there's a gap, LoRA is part of the problem and full fine-tuning (or higher rank / different target modules) should be considered for SFT.

### 3.5 LoRA Diagnostic

Run base model eval in two modes:
1. **Raw base model** — `AutoModelForCausalLM.from_pretrained()` directly
2. **Base model + untrained LoRA** — same LoRA config as training (r=128, alpha=256), adapter weights at init (should be identity/zero)

If (1) ≈ (2): LoRA is not the issue, training dynamics are.
If (1) >> (2): LoRA itself degrades the base model, even before training. Consider full fine-tuning for SFT.

### 3.6 Expected Output

```
eval_results/base_qwen_0.5b_full.json
```

With metrics:
- `gsm8k_accuracy` (target: ~44%)
- `math_accuracy` (by level)
- `avg_tokens_easy`, `avg_tokens_hard`
- Per-problem results for error analysis

---

## 4. Task B — SFT Fine-Tuning

### 4.1 What to Build

A script `scripts/training/train_sft.py` that:
- Loads `Qwen/Qwen2.5-0.5B` with LoRA
- Performs standard supervised fine-tuning (next-token prediction on chosen solutions)
- Uses the same dataset as DPO (balanced_v4_capped) but only the `chosen` responses
- Supports same CLI args as existing training scripts (lr, epochs, batch_size, etc.)
- Logs to wandb
- Saves checkpoints

### 4.2 Training Data

From `data/processed_dpo_dataset_balanced_v4_capped/train.jsonl`:
- Each line has `problem`, `chosen`, `rejected`, `complexity`
- For SFT, we only use `problem` + `chosen` (the correct solution)
- Format each example as: `"Problem: {problem}\nSolution: {chosen}"` and train on the solution tokens only (mask the prompt)
- Use the same train/val split

### 4.3 SFT Loss

Standard cross-entropy on solution tokens. No DPO, no preference pairs, no length penalties.

### 4.4 Hyperparameter Sweep (Autoresearch Loop)

Start with reasonable defaults, then iterate:

| Parameter | Starting Value | Sweep Range |
|-----------|---------------|-------------|
| lr | 2e-5 | 1e-5, 2e-5, 5e-5 |
| epochs | 3 | 1, 2, 3, 5 |
| batch_size | 4 | 4, 8 |
| LoRA rank | 128 | 64, 128 |
| LoRA alpha | 256 | 128, 256 |
| gradient_accumulation | 1 | 1, 2, 4 |
| warmup_ratio | 0.1 | 0, 0.05, 0.1 |
| weight_decay | 0.01 | 0, 0.01 |

### 4.5 Evaluation

After each SFT run, evaluate with the same pipeline:
```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/<sft_name>/best-model \
  --output eval_results/<sft_name>.json --use-real --limit 500
```

Once we have a promising model, do a full eval (no --limit) for final numbers.

### 4.6 Expected Output

- `checkpoints/sft_<name>/` — model checkpoints
- `eval_results/sft_<name>.json` — eval results
- `autoresearch/iteration11.md`, `iteration12.md`, ... — iteration docs

---

## 5. Iteration Protocol (Same as Before)

Each iteration produces `autoresearch/iterationN.md` with:
1. Hypothesis
2. Hyperparameters (full CLI invocation)
3. Changes made
4. Results (epoch-level metrics + post-train eval)
5. Comparison to baseline (base model) and previous best
6. Analysis
7. Next iteration plan

Continue numbering from iteration 11.

---

## 6. Success Criteria

| Milestone | Criteria |
|-----------|---------|
| Base model evaluated | We know exact GSM8K accuracy on our pipeline |
| Gap understood | We can explain any difference from published 44% |
| SFT beats base | SFT accuracy > base model accuracy on easy problems |
| SFT accuracy maximized | At least 3 iterations of hyperparameter tuning, diminishing returns |
| All documented | Every run in results.tsv, every decision in iteration docs |

---

## 7. What NOT to Do

- Do NOT optimize for token count
- Do NOT use DPO loss (use standard SFT cross-entropy)
- Do NOT run budget-aware anything
- Do NOT run multiple GPUs (only 1 available)
- Do NOT skip the base model evaluation — it's the foundation for everything

---

## 8. Execution Order

1. **Write `scripts/eval_base_model.py`** — base model eval script
2. **Run base model eval** (full dataset, max_new_tokens=256, then 512)
3. **Document as iteration 11** — base model baseline
4. **Write `scripts/training/train_sft.py`** — SFT training script
5. **Run first SFT experiment** with default hyperparameters
6. **Evaluate and document as iteration 12**
7. **Iterate** — tune hyperparameters based on results
8. **Continue until accuracy plateaus** or user interrupts

---

## 9. DPO Loss Design Notes

### 9.1 Length Normalization in `log_prob`

The `log_prob` function in `src/training/dpo_trainer.py` (line 271) computes **mean per-token log-probability**, not the sum:

```python
return (token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
```

This is intentional and correct for budget-aware DPO. Here is why.

**Standard DPO theory** uses sum log-probs: `log π(y|x) = Σ_t log p(y_t | y_{<t}, x)`. The sum grows with sequence length, which means longer responses naturally have a more negative total log-probability. This creates an implicit length bias in the DPO reward that is hard to control.

**Mean log-probs** remove that implicit length effect: the quality term in the reward reflects average per-token quality regardless of sequence length. Length preference is then expressed **explicitly** via the dedicated length penalty in the budget-aware loss:

```
reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)   # quality, O(1)
            - lambda * (chosen_len - rejected_len) / avg_len   # length, O(1)
```

Both terms are O(1) because both the DPO quality term (mean log-prob) and the length penalty (normalized by avg_len) are length-agnostic in scale. This makes `beta` and `lambda` interpretable hyperparameters that do not need to be rescaled as average sequence length changes. Using sum log-probs would make the quality term O(|y|), putting it on a completely different scale than the fixed-size length penalty.

**Sign check**: chosen is shorter → `chosen_len - rejected_len < 0` → `length_penalty < 0` → subtracting it increases `reward_diff` → loss decreases → model prefers shorter correct solution. ✓

### 9.2 Code Fix: Response-Only Log-Prob Masking

**Current behaviour**: `log_prob` averages over all non-padding tokens, including prompt tokens (which are identical for chosen and rejected pairs). Prompt tokens introduce a systematic bias: shorter responses give the prompt a higher weight in the mean, which slightly inflates `log_ratio_chosen` if the policy is improving on the prompt.

**Fix**: mask out prompt tokens and normalize only over response tokens. Requires:

1. **Preprocessing** (`src/data/preprocessing.py`): store `chosen_prompt_length` and `rejected_prompt_length` as integer tensors alongside `chosen_input_ids` in the `.pt` file.

2. **Dataset** (`TokenizedDPODataset.__getitem__`): include `chosen_prompt_length` and `rejected_prompt_length` in the returned dict.

3. **`log_prob`** signature change:
   ```python
   def log_prob(logits, input_ids, attention_mask, prompt_length: Optional[int] = None):
       shift_mask = attention_mask[..., 1:].contiguous().float()
       if prompt_length is not None:
           shift_mask[..., :prompt_length] = 0.0   # zero out prompt positions
       ...
       return (token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
   ```

4. **`_compute_batch_forward`**: pass `chosen_prompt_length` and `rejected_prompt_length` from batch to the two `log_prob` calls.

This fix is low-risk (the prompt bias is small in practice due to LoRA), but it makes the implementation cleaner and theoretically correct. It is not required before budget-aware DPO experiments resume, but should be done before any camera-ready or ablation runs.
