# HPO v6 Scan Results
**Study:** `budget_dpo_hpo_0504_150740`  
**Dates:** 2026-05-04 → 2026-05-06 (ongoing)  
**Log:** `logs/hpo_run_v6.log`  
**DB:** `checkpoints/optuna/budget_dpo_hpo_0504_150740.db`

---

## 1. Study Configuration

| Field | Value |
|-------|-------|
| Sampler | TPE (`multivariate=True, group=True`) |
| Pruner | NopPruner |
| Objective | `efficiency` = accuracy / (mean_gen_length / 1024), higher is better |
| Accuracy floor | 0.15 (infeasible if acc_easy below this) |
| Max epochs | 3 |
| Train size | 1,000 unique problems |
| Val size | 250 unique problems |
| Max seq len | 1,536 tokens |
| Val gen batch size | 8 |
| Model | `Qwen/Qwen2.5-Math-1.5B` |
| Dataset | `data/processed_dpo_dataset_full/` |
| Total planned | 20 trials |

### Fixed parameters (not sampled)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `loss_type` | dpo | SimPO collapsed accuracy in all v5 trials |
| `kl_penalty_weight` | 0.0 | High KL froze policy in v5 |
| `batch_size` | 8 | batch=16 always OOMed in v5 |

### TPE search space (v6)

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `lr` | log-uniform | 4e-7 – 2e-6 |
| `dpo_beta` | log-uniform | 0.08 – 0.25 |
| `lambda_easy` | log-uniform | 0.10 – 0.35 |
| `lambda_hard` | log-uniform | 1e-4 – 5e-4 |
| `gradient_accumulation_steps` | categorical | {1, 2} |
| `length_ratio_easy` | uniform | 1.5 – 3.0 |
| `length_ratio_hard` | uniform | 2.0 – 3.0 |
| `max_pairs_per_problem` | int-uniform | 10 – 20 |

### Launch command

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/storage/arik/nlp_final_project \
PYTHONUNBUFFERED=1 \
nohup .venv/bin/python -m scripts.optuna_hpo \
  --n-trials 20 \
  --max-epochs 3 \
  --train-size 1000 \
  --val-size 250 \
  --objective efficiency \
  --accuracy-floor 0.15 \
  --max-seq-len 1536 \
  --val-gen-batch-size 8 \
  --sampler tpe \
  --wandb \
  > logs/hpo_run_v6.log 2>&1 &
```

---

## 2. All Trial Results

**Status as of 2026-05-06 ~09:00:** 15 trials complete, trial 15 running (collapsed mid-run), trials 16–19 pending.

### 2.1 Final-epoch results, sorted by efficiency

| # | Eff↓ | Overall acc | Easy acc | Hard acc | Easy tokens | val_loss@E3 | Regime |
|---|------|------------|---------|---------|------------|-------------|--------|
| **10** | **2.923** | 0.272 | 0.280 | 0.264 | 83 | 0.150 | collapsed |
| 14 | 2.542 | 0.256 | 0.240 | 0.272 | 109 | 0.101 | collapsed |
| 13 | 2.290 | 0.276 | 0.288 | 0.264 | 125 | 0.135 | collapsed |
| 3 | 2.122 | 0.252 | 0.256 | 0.248 | 147 | 0.166 | partial |
| 11 | 2.011 | 0.260 | 0.296 | 0.224 | 102 | 0.131 | collapsed |
| 0 | 1.747 | 0.248 | 0.288 | 0.208 | 143 | 0.083 | collapsed |
| 5 | 1.506 | 0.288 | 0.328 | 0.248 | 150 | 0.597 | **stable** |
| 9 | 1.440 | 0.292 | 0.336 | 0.248 | 146 | 0.641 | **stable** |
| 6 | 1.348 | 0.352 | 0.416 | 0.288 | 187 | 0.643 | **stable** |
| 12 | 1.142 | 0.356 | 0.432 | 0.280 | 231 | 0.676 | **stable** |
| 4 | 1.104 | 0.304 | 0.328 | 0.280 | 211 | 0.630 | **stable** |
| 1 | 1.024 | 0.244 | 0.232 | 0.256 | 307 | 0.100 | degenerate |
| 7 | 0.971 | 0.332 | 0.408 | 0.256 | 247 | 0.663 | **stable** |
| 2 | 0.962 | 0.348 | 0.408 | 0.288 | 250 | 0.666 | **stable** |
| 8 | 0.889 | 0.348 | 0.424 | 0.272 | 295 | 0.662 | **stable** |
| 15 | — | — | — | — | — | 0.059@E2 | **crashed** |

### 2.2 Full hyperparameters for all completed trials

| # | lr | beta | λ_easy | λ_hard | accum | ratio_easy | ratio_hard | max_pairs |
|---|----|------|--------|--------|-------|-----------|-----------|-----------|
| 0 | 7.31e-7 | 0.236 | 0.250 | 2.62e-4 | 1 | 1.587 | 2.866 | 16 |
| 1 | 1.25e-6 | 0.082 | 0.337 | 3.82e-4 | 1 | 1.775 | 2.304 | 15 |
| 2 | 8.02e-7 | 0.111 | 0.215 | 1.25e-4 | 2 | 2.184 | 2.785 | 12 |
| 3 | 9.15e-7 | 0.157 | 0.106 | 2.66e-4 | 1 | 2.923 | 2.966 | 18 |
| 4 | 6.53e-7 | 0.089 | 0.236 | 2.03e-4 | 2 | 1.552 | 2.909 | 12 |
| 5 | 1.16e-6 | 0.114 | 0.192 | 2.41e-4 | 2 | 2.663 | 2.939 | 19 |
| 6 | 1.05e-6 | 0.229 | 0.112 | 1.37e-4 | 2 | 2.083 | 2.271 | 19 |
| 7 | 7.10e-7 | 0.110 | 0.197 | 1.25e-4 | 1 | 2.980 | 2.772 | 12 |
| 8 | 4.04e-7 | 0.203 | 0.242 | 3.23e-4 | 1 | 2.038 | 2.116 | 19 |
| 9 | 1.09e-6 | 0.117 | 0.108 | 1.65e-4 | 2 | 2.456 | 2.887 | 15 |
| 10 | 1.08e-6 | 0.164 | 0.124 | 2.01e-4 | 1 | 2.872 | 2.959 | 20 |
| 11 | 9.80e-7 | 0.177 | 0.139 | 2.60e-4 | 1 | 2.774 | 2.886 | 20 |
| 12 | 5.42e-7 | 0.156 | 0.103 | 1.94e-4 | 1 | 2.361 | 2.529 | 16 |
| 13 | 8.43e-7 | 0.220 | 0.119 | 3.29e-4 | 1 | 2.650 | 2.944 | 14 |
| 14 | 8.83e-7 | 0.234 | 0.113 | 3.23e-4 | 1 | 2.367 | 2.632 | 14 |
| 15 | 1.21e-6 | 0.222 | 0.140 | 3.80e-4 | 1 | 1.669 | 2.551 | 14 |

### 2.3 Epoch-by-epoch trajectories

Format: `acc_overall (easy_acc) | mean_tokens | efficiency`

| # | Epoch 1 | Epoch 2 | Epoch 3 | val_loss@E3 | Note |
|---|---------|---------|---------|-------------|------|
| 0 | 0.400 (0.488) \| 464 \| 0.88 | 0.276 (0.320) \| 181 \| 1.56 | 0.248 (0.288) \| 145 \| 1.75 | 0.083 | steady collapse E2→E3 |
| 1 | 0.336 (0.424) \| 435 \| 0.79 | 0.256 (0.240) \| 262 \| 1.00 | 0.244 (0.232) \| 244 \| 1.02 | 0.100 | degenerate: tokens UP, accuracy DOWN |
| 2 | 0.312 (0.424) \| 465 \| 0.69 | 0.352 (0.432) \| 409 \| 0.88 | 0.348 (0.408) \| 370 \| 0.96 | 0.666 | stable, slow decline |
| 3 | 0.324 (0.416) \| 437 \| 0.76 | 0.364 (0.432) \| 359 \| 1.04 | 0.252 (0.256) \| 122 \| 2.12 | 0.166 | good E2, cliff drop E3 |
| 4 | 0.336 (0.408) \| 479 \| 0.72 | 0.340 (0.392) \| 469 \| 0.74 | 0.304 (0.328) \| 282 \| 1.10 | 0.630 | slow but stable |
| 5 | 0.320 (0.400) \| 441 \| 0.74 | 0.348 (0.424) \| 389 \| 0.92 | 0.288 (0.328) \| 196 \| 1.51 | 0.597 | **stable, good balance** |
| 6 | 0.336 (0.440) \| 443 \| 0.78 | 0.348 (0.440) \| 394 \| 0.90 | 0.352 (0.416) \| 268 \| 1.35 | 0.643 | **stable, high accuracy** |
| 7 | 0.364 (0.440) \| 451 \| 0.83 | 0.316 (0.376) \| 387 \| 0.84 | 0.332 (0.408) \| 350 \| 0.97 | 0.663 | stable but slow token reduction |
| 8 | 0.336 (0.448) \| 452 \| 0.76 | 0.320 (0.400) \| 430 \| 0.76 | 0.348 (0.424) \| 401 \| 0.89 | 0.662 | stable but very slow token reduction |
| 9 | 0.320 (0.392) \| 447 \| 0.73 | 0.344 (0.416) \| 373 \| 0.94 | 0.292 (0.336) \| 208 \| 1.44 | 0.641 | **stable, good balance** |
| 10 | 0.328 (0.440) \| 440 \| 0.76 | 0.324 (0.384) \| 256 \| 1.30 | 0.272 (0.280) \| 95 \| 2.92 | 0.150 | collapse E3, big token drop |
| 11 | 0.352 (0.440) \| 442 \| 0.81 | 0.360 (0.416) \| 351 \| 1.05 | 0.260 (0.296) \| 132 \| 2.01 | 0.131 | good E2, collapse E3 |
| 12 | 0.328 (0.432) \| 445 \| 0.75 | 0.324 (0.408) \| 408 \| 0.81 | 0.356 (0.432) \| 319 \| 1.14 | 0.676 | stable, improving E3 |
| 13 | 0.324 (0.384) \| 433 \| 0.77 | 0.380 (0.448) \| 344 \| 1.13 | 0.276 (0.288) \| 123 \| 2.29 | 0.135 | **best E2 accuracy (0.448)**, collapse E3 |
| 14 | 0.348 (0.464) \| 418 \| 0.85 | 0.336 (0.400) \| 306 \| 1.13 | 0.256 (0.240) \| 103 \| 2.54 | 0.101 | collapse E3, worst accuracy |
| 15 | 0.368 (0.464) \| 369 \| 1.02 | 0.256 (0.304) \| 411 \| 0.64 | running | 0.059@E2 | catastrophic E2 collapse: tokens UP, reward_diff=4.84 |

---

## 3. The Two-Regime Pattern

The 15 completed trials split into two qualitatively distinct groups.

### 3.1 Stable regime (8 trials)

Trials 2, 4, 5, 6, 7, 8, 9, 12.  
- val_loss at epoch 3: 0.597–0.676 (DPO loss behaving normally)
- Accuracy holds at 33–44% easy across all 3 epochs
- Token reduction is gradual: starts ~440–480, ends 196–401
- No reward divergence

### 3.2 Collapsed regime (6 trials)

Trials 0, 3, 10, 11, 13, 14.  
- val_loss at epoch 3: 0.083–0.166 (reward divergence, loss → 0)
- Epoch 1 looks normal: easy_acc 40–49%
- Epoch 3: easy_acc drops to 24–30%, tokens compress to 83–147
- The efficiency gain is artificial: the model's reward diverges (policy log-ratio → ∞), which mechanically compresses outputs but simultaneously tanks accuracy

### 3.3 Degenerate case (trial 1)

- Collapsed (val_loss=0.100) but tokens went UP (avg_easy=307)
- Easy accuracy fell to 23.2%
- Failure mode: collapse + length increase simultaneously, possibly repetitive/padding outputs

### 3.4 Catastrophic early collapse (trial 15, running)

- Epoch 2: reward_diff=4.84 (normal range is 0.03–0.30), val_loss=0.059, tokens_easy=475 (longer!)
- Cause: lr=1.21e-6 (highest in study) + length_ratio_easy=1.669 (lowest of high-efficiency cluster) → fast divergence in the wrong direction
- Expected final score: poor

### 3.5 What separates the regimes

The collapsed trials are NOT distinguished by λ_easy alone. Several stable trials use λ_easy ≥ 0.10 (trials 5, 9). The collapse is determined by the interaction of lr + max_pairs + length_ratio_easy creating a dataset where early reward gradient is steep enough to cause divergence by epoch 3. Specifically, trials 10, 11, 13, 14 share: `accum=1 + ratio_easy ≥ 2.4 + ratio_hard ≥ 2.6 + lr ~0.8–1.1e-6` with `max_pairs ≥ 14`.

---

## 4. Research Question Answers

### RQ1 — The accuracy–token tradeoff crossing point (PRIMARY)
**Question:** Can we simultaneously achieve easy_acc ≥ 0.30 AND easy_tokens ≤ 120?

**Answer: No, not in 15 trials.**

Pareto frontier observed:

| easy_tokens range | best easy_acc achievable |
|-------------------|-------------------------|
| ≤ 110 | 24–30% (trials 10, 11, 14) |
| 110–150 | 29–34% (trials 0, 5, 9) |
| 150–200 | 33–42% (trials 5, 6) |
| 200+ | 41–44% (trials 7, 8, 12) |

Trial 11 is the closest: easy_acc=29.6%, tokens=102. Still ~0.5% short of the threshold.

The difficulty is structural: the λ_easy values that get tokens below 120 (≥ 0.11 in the collapsed configs) cause DPO reward divergence that tanks accuracy by epoch 3. At shorter training (e.g. epoch 2, before collapse), the accuracy is high but tokens are not yet that short.

**Implication for deep training:** Checkpoint selection at mid-training (before collapse) is critical. The `best_model_metric="val_loss"` bug (see §6) means we are currently saving the most-collapsed checkpoint, not the best one.

### RQ2 — Is max_pairs ≥ 13 a real causal floor?
**Answer: Partial support, but confounded.**

Trials with max_pairs = 12 (trials 2, 4, 7): all stable, efficiency 0.89–1.10.  
Trials with max_pairs ≥ 14 in the same accum=1 cluster (10, 11, 13, 14): high efficiency but collapsed.  
The low-max_pairs trials were also paired with different lr/accum values, so isolation is not clean.

Preliminary conclusion: max_pairs ≥ 14 alone does not cause collapse, but it may amplify it. **Not yet conclusive.**

### RQ3 — Does dpo_beta matter?
**Answer: No. Beta is a don't-care in [0.08, 0.25].**

High-efficiency cluster: beta = 0.164 (T10), 0.220 (T13), 0.234 (T14), 0.177 (T11). No pattern.  
Best stable trials: beta = 0.114 (T5), 0.229 (T6), 0.117 (T9). No pattern.  
**Recommendation for deep training: fix beta = 0.15.**

### RQ4 — Where does length_ratio_hard saturate?
**Answer: ≥ 2.5 consistently better; likely plateaus around 2.7–2.9.**

Trial 6 (ratio_hard=2.27): efficiency=1.35 — good accuracy (42%) but lower efficiency.  
Trial 8 (ratio_hard=2.12): efficiency=0.89 — lowest in the study.  
All top-5 trials: ratio_hard ≥ 2.63.  
No evidence that 3.0 beats 2.8 within the top cluster — both appear in top trials.  
**Recommendation: fix ratio_hard = 2.9 for deep training.**

### RQ5 — Does grad_accum=2 ever beat grad_accum=1?
**Answer: No. grad_accum=1 wins clearly.**

Trials with accum=2: 2, 4, 5, 6, 9. Max efficiency = 1.51 (trial 5).  
Trials with accum=1: 0, 1, 3, 7, 8, 10, 11, 12, 13, 14. Top 6 by efficiency all use accum=1.  
However: the accum=2 trials are disproportionately in the stable regime (all 5 are stable). Whether accum=2 *prevents* collapse or whether the stable accum=2 trials just happened to land in safe configs is unclear.  
**Recommendation: use accum=1 for deep training (higher efficiency), but if collapse is a concern use accum=2 as a stabilizer.**

### RQ6 — Is λ_hard a don't-care within [1e-4, 5e-4]?
**Answer: Yes, effectively a don't-care.**

Top-efficiency trials span the full range: 2.01e-4 (T10), 2.60e-4 (T11), 3.29e-4 (T13), 3.23e-4 (T14).  
Best stable trial (T6): 1.37e-4. Best stable balance (T9): 1.65e-4.  
No monotonic relationship visible.  
**Recommendation: fix λ_hard = 1e-4 for deep training (minimum of range).**

---

## 5. Full Test Set Evaluation — Trial 10

Trial 10 was evaluated on the full test set (`eval_results/optuna_trial10_zeroShot.json`).

| Metric | HPO val (250 problems) | Full test (6,319 problems) |
|--------|----------------------|---------------------------|
| Overall accuracy | 27.2% | **20.4%** |
| Easy accuracy | 28.0% | **29.7%** |
| Hard accuracy | 26.4% | **16.8%** |
| Avg easy tokens | 83 | **79** |
| Avg hard tokens | — | **143** |
| Efficiency | 2.923 | **1.664** |

The HPO val set was 33% optimistic on overall accuracy. Hard accuracy degrades substantially on the full test set (26% → 17%), which is expected — 250 hard problems is a noisy sample of MATH Level 4–5. The easy token count is consistent (83 vs 79).

**Math by level (full eval):**
| Level | Accuracy |
|-------|----------|
| 1 | 45.3% |
| 2 | 31.2% |
| 3 | 22.0% |
| 4 | 12.9% |
| 5 | 6.0% |

The model retains reasonable level 1–2 ability but essentially cannot solve level 4–5 (9.3% on levels 4+5). This is consistent with the collapse — token budget was cut too aggressively.

---

## 6. Known Bug: Checkpoint Selector

**File:** `scripts/optuna_hpo.py` line 255  
**Problem:** `best_model_metric="val_loss"` is hardcoded. All saved `best-model` checkpoints are selected by lowest val_loss, which is the DPO collapse signature. For collapsed trials (10, 11, 13, 14), the saved checkpoint is from the *most* diverged epoch.

**Fix (commented in-place for manual activation):**
```python
# current (saves most-collapsed epoch):
best_model_metric="val_loss",  # TODO: best_model_metric="efficiency", accuracy_floor=search.accuracy_floor,
accuracy_floor=None,
```

**Impact on HPO scores:** The HPO score is computed from `best_metrics` loaded from the saved checkpoint file. Since val_loss selects epoch 3 for most trials (val_loss monotonically decreases), and the efficiency score is also from epoch 3, the ranking is still correct. The problem is that the saved `best-model` artifact is the collapsed state — **do not use these checkpoints as init points for deep training without selecting the right epoch manually.**

---

## 7. Best Configs for Deep Training

### Selection criterion
Using the balanced criterion from the session plan: `easy_acc ≥ 0.30 AND easy_tokens ≤ 130`. No trial from v6 satisfies both simultaneously. Falling back to: top 3 from **stable regime** (genuine learning, val_loss ≥ 0.59), ranked by efficiency.

### Top 3 stable-regime candidates

**Candidate A — Trial 9** (efficiency=1.44, balance pick)
```
lr=1.09e-6, dpo_beta=0.117, lambda_easy=0.108, lambda_hard=1.65e-4
grad_accum=2, length_ratio_easy=2.456, length_ratio_hard=2.887, max_pairs=15
E3: easy_acc=33.6%, easy_tokens=146, overall_acc=29.2%
```

**Candidate B — Trial 5** (efficiency=1.51, slightly higher efficiency than A)
```
lr=1.16e-6, dpo_beta=0.114, lambda_easy=0.192, lambda_hard=2.41e-4
grad_accum=2, length_ratio_easy=2.663, length_ratio_hard=2.939, max_pairs=19
E3: easy_acc=32.8%, easy_tokens=150, overall_acc=28.8%
```

**Candidate C — Trial 6** (highest accuracy, moderate token reduction)
```
lr=1.05e-6, dpo_beta=0.229, lambda_easy=0.112, lambda_hard=1.37e-4
grad_accum=2, length_ratio_easy=2.083, length_ratio_hard=2.271, max_pairs=19
E3: easy_acc=41.6%, easy_tokens=187, overall_acc=35.2%
```

### Recommendations before launching deep runs

1. **Fix the checkpoint selector** (uncomment the TODO on line 255 of `optuna_hpo.py`) when starting the next HPO scan or deep training.
2. Use `--max-epochs 8` (vs 3 in HPO) — stable configs should continue improving; the plateau is not yet reached.
3. Use `--best-model-metric efficiency --accuracy-floor 0.20` to catch the best pre-collapse epoch.
4. Fix `beta=0.15`, `lambda_hard=1e-4` as per RQ answers above to reduce noise.

### Deep training command template

```bash
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=data/processed_dpo_dataset_full \
PYTHONPATH=/storage/arik/nlp_final_project \
PYTHONUNBUFFERED=1 \
nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/deep_run_A \
  --max-epochs 8 \
  --batch-size 8 \
  --lr 1.09e-6 \
  --dpo-beta 0.117 \
  --lambda-easy 0.108 \
  --lambda-hard 1e-4 \
  --kl-penalty 0.0 \
  --gradient-accumulation-steps 2 \
  --best-model-metric efficiency \
  --accuracy-floor 0.20 \
  --run-name deep_run_A \
  --wandb \
  > logs/deep_run_A.log 2>&1 &
```

---

## 8. What Trials 16–19 Should Clarify

With 4 trials remaining, TPE will likely continue sampling near the high-efficiency cluster (lr ~0.8–1.1e-6, λ_easy ~0.11–0.14, ratio_easy ~2.7–3.0, ratio_hard ~2.9, accum=1). The main open questions:

1. **Can the collapse be avoided while keeping tokens ≤ 120?** — If TPE samples a config with accum=2 in the high-efficiency region, it may find a stable point with both short outputs and decent accuracy.

2. **Does λ_easy = 0.14–0.19 with accum=1 produce a new stable-but-efficient point?** — Trials 5 and 9 used accum=2 with λ_easy in this range and got tokens 146–150. With accum=1 it might push lower.

3. **RQ2 resolution** — If any new trial uses max_pairs=10–12 in the otherwise high-efficiency cluster, we can finally isolate the max_pairs effect.

These remaining trials don't change the deep training recommendations above — those configs are already identified. The remaining trials are primarily informational.

---

## 9. Comparison with v5 Scan

| Metric | v5 best (trial 11) | v6 best by efficiency (trial 10) | v6 best stable (trial 6) |
|--------|-------------------|----------------------------------|--------------------------|
| Efficiency | 1.987 | **2.923** | 1.348 |
| Easy accuracy | 31.4% | 28.0% | **41.6%** |
| Easy tokens | 83 | 83 | 187 |
| Hard accuracy | 27.6% | 26.4% | **28.8%** |
| val_loss | (unknown) | 0.150 | 0.643 |
| Regime | collapsed | collapsed | stable |

v6 found higher peak efficiency but did not improve on the accuracy-token tradeoff. The RQ1 crossing point (≥30% acc, ≤120 tokens) remains unsolved. v6 confirmed that the mechanism for high efficiency is DPO collapse, not genuine length learning.

The highest efficiency in the stable regime increased from v5 (no comparable stable trial logged) to v6's 1.51 (trial 5) — this is a genuine improvement, as the v5 equivalent stable trials clustered around efficiency 0.7–0.9 due to the wider search space including KL penalty and SimPO.
