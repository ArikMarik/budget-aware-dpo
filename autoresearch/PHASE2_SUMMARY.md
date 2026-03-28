# Budget-Aware DPO — Phase 2 Summary & Phase 3 Handoff

**Date**: 2026-03-28
**Status**: Phase 2 complete. Pivoting to Phase 3.
**Branch**: `autoresearch/mar26`

---

## 1. What We Tested in Phase 2

Phase 2 had 3 GPUs and ~12 hours. We tested:
- **Hypothesis B**: Larger model (Qwen2.5-1.5B) — does more capacity amplify the budget-aware effect?
- **Hypothesis D**: SimPO loss — does built-in length normalization help?
- **Hypothesis E**: Two-phase training — warmup for accuracy, then add length penalty

## 2. Iteration Index

| Iter | Name | Key Idea | Result |
|------|------|----------|--------|
| 0 | Lambda too small | λ=0.05 on 0.5B | No divergence from baseline |
| 1 | Strong lambda | λ=5.0 on 0.5B | Accuracy up but tokens 2x LONGER |
| 2 | Data fix + gen-eval | Capped data, gen eval | Correct direction E1 but 1% accuracy |
| 3 | Large diverse data | 51K real data | Model collapse |
| 4 | Lower LR + KL | lr=1e-6, KL=0.1 | KL too strong, model frozen |
| 5 | Tuned KL | KL=0.01 | **Phase 1 best**: 22% post-train, 177 easy tok |
| 6 | Fair baseline | Baseline + KL=0.01 | 21.2% post-train, 179 easy tok |
| 7 | 1.5B model + SimPO | 3 parallel experiments | 1.5B budget 25.8% post-train; SimPO failed |
| 8 | Two-phase 0.5B | DPO warmup → budget | 25-30% gen-eval, not better than 1.5B |
| 9 | 1.5B baseline no-KL | Fix bad baseline | Crashed (GPU memory leak) |

## 3. Phase 2 Results — Post-Training Eval (Ground Truth)

| Model | Overall Acc | Easy Acc | Hard Acc | Easy Tok | Hard Tok | TPCA | MATH L4-5 |
|-------|-----------|---------|---------|---------|---------|------|-----------|
| 0.5B Baseline (iter6) | 21.2% | 27.6% | 14.8% | **179** | 198 | 891 | 11.6% |
| 0.5B Budget (iter5) | 22.0% | **29.2%** | 14.8% | **177** | 195 | 846 | 8.2% |
| 1.5B Budget E1 | 24.6% | 22.4% | 26.8% | 244 | 177 | 856 | **13.7%** |
| 1.5B Budget E2 | **25.8%** | 24.0% | **27.6%** | 240 | 179 | **812** | **13.7%** |

## 4. Key Findings

### 4.1 The 0.5B Model is Better on Easy Problems
Counterintuitively, the 0.5B model outperforms 1.5B on easy problems (29.2% vs 24.0%). The 1.5B model wins on hard problems (27.6% vs 14.8%) but generates much longer easy responses (240 vs 177 tokens). The budget-aware token shortening effect on easy problems **does not generalize from training to held-out data** on the 1.5B model.

### 4.2 SimPO Doesn't Work
SimPO (no reference model) overfits catastrophically. Both β=2.0 and β=0.5 led to train loss near 0 with terrible val performance. The reference model is essential for stability.

### 4.3 Two-Phase Training Works but Doesn't Beat Direct Training
Two-phase (DPO warmup → budget DPO) on 0.5B achieved 25-30% gen-eval accuracy but doesn't beat the 1.5B budget model or even match the 0.5B budget model on held-out data.

### 4.4 In-Training vs Held-Out Gap Remains Large
1.5B budget E2: 43% in-training → 25.8% held-out. This ~17 point gap is consistent across all models. In-training gen-eval is useful for relative comparison but not for final numbers.

### 4.5 1.5B Baseline with KL=0.01 Completely Failed
Only 5-6% accuracy across 2 epochs. The KL penalty without length penalty over-constrains the 1.5B model. The budget-aware lambda somehow helps the model learn faster — an interesting finding.

## 5. Phase 3 Direction: All-In on 0.5B Easy Problems

### Strategy
The 0.5B model already gets 29.2% on easy (GSM8K) problems with 177 tokens. Base Qwen2.5-0.5B gets ~36% on GSM8K from benchmarks. The ceiling is probably 40-50% with optimized DPO training.

**Goal**: Push 0.5B easy accuracy as high as possible (target: 35-45%), while making budget-aware responses **shorter** than baseline on easy problems, and maintaining hard accuracy.

### Approach
1. **Go all-in on 0.5B**: The larger model doesn't help on easy problems
2. **Use hard problems as classifier training**: Easy = be as concise as possible, Hard = don't touch reasoning
3. **Push both baseline and budget-aware**: Get the best easy accuracy we can, then show budget achieves same accuracy with fewer tokens
4. **Hyperparameter search**: lambda (5, 10, 20, 50), epochs (3, 6, 10), KL (0, 0.01, 0.005), lr (1e-6, 5e-6)
5. **Data strategy**: Consider oversampling easy problems, or adjusting the easy/hard classification threshold

### Experiments to Run
- **0.5B Budget with more epochs** (6-10 epochs, λ=5.0, KL=0.01) — does longer training help?
- **0.5B Budget with stronger lambda** (λ=10, λ=20) — does more penalty = shorter easy?
- **0.5B Baseline with more epochs** (6-10 epochs, KL=0.01) — what's the accuracy ceiling?
- **0.5B Budget without KL but with warmup** — test if warmup replaces KL
- **Different easy/hard threshold** — currently GSM8K=easy, MATH L4-5=hard. Try including MATH L1-2 as easy.

### Success Criteria
| Metric | Target |
|--------|--------|
| Easy accuracy (held-out) | ≥35% (currently 29.2%) |
| Easy tokens budget | <160 (currently 177) |
| Easy tokens budget vs baseline | Budget ≤ 90% of baseline |
| Hard accuracy | ≥ baseline (currently 14.8%) |
| TPCA | Budget < baseline |

### GPU Assignment
- GPU 0: stuck with memory leak — may need process restart
- GPU 1: Available (49GB)
- GPU 2: Available after eval finishes (49GB)

Run 0.5B models (8GB each) — can fit 2 experiments per GPU if needed.
