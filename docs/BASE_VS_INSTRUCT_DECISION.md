# Model Selection: Qwen2.5-Math-1.5B (Base) vs Instruct

**Decision**: Use `Qwen2.5-Math-1.5B` (base model). Do not use the Instruct variant.

---

## What the Instruct Variant Received

The instruct model underwent three post-training stages beyond the base:

1. **SFT with rejection sampling** — iterative selection of high-quality CoT paths ranked by correctness + reward score
2. **Reward model training** — scalar value head trained on 361K EN + 257K ZH problems
3. **GRPO** — reinforcement learning that explicitly rewards structured, thorough solutions with `\boxed{}` answers and intermediate reasoning steps

These stages baked in strong output-format priors: verbose chain-of-thought, boxed answers, tool-integrated reasoning (TIR). The instruct model was trained to treat "long reasoning = correct."

---

## Why Instruct + Budget-Aware DPO Is a Bad Combination

Budget-aware DPO works by shifting the model's output distribution toward shorter solutions on easy problems. When the starting model was GRPO-aligned to be verbose, the DPO gradient is fighting that prior at every step:

- The frozen reference model assigns high log-probability to long, structured outputs
- DPO pushes the policy toward shorter outputs
- The KL term resists this shift — training instability, high gradient norms, potential collapse

This project already observed these dynamics (iterations 0–4) on the 0.5B model when lambda/KL were mistuned. Starting from an instruct model would make the landscape harder to navigate, not easier. The base model has no such format priors: it is a blank slate for the DPO signal to shape.

**Official guidance**: The Qwen team designates base models for downstream fine-tuning; instruct models are designated "for chatting."

---

## The SFT Cold-Start (Is It Worth It?)

A Base → SFT → DPO pipeline (used by DeepSeek-R1, Qwen's own post-training) would:
- Install format conventions (zero-shot, `\boxed{}` answers) via a cheap 1–2 epoch SFT on chosen solutions
- Give DPO a better-aligned reference distribution to work against
- Eliminate any risk of the base model producing unstructured outputs during DPO training

**Project context**: SFT experiments were already run in Phase 4 (iterations 11–12, `scripts/training/train_sft.py`). The project then moved to direct DPO on the base model, which is producing real signal: 25.8% accuracy and 811.5 TPCA on 1.5B (iter7b E2). The base model is already producing parseable answers — the SFT cold-start would offer marginal improvement at the cost of an additional training phase.

| Strategy | Alignment Tax | Format Risk | Complexity |
|---|---|---|---|
| Instruct + DPO | High — GRPO priors resist length changes | None | Low |
| **Base + DPO (current)** | **Low** | **Low (already working)** | **Medium** |
| Base + SFT + DPO | Low | None | High |

---

## On the 8-Shot Evaluation Protocol

8-shot exemplars are used **at evaluation time only**, never during training. Training pairs are formatted zero-shot:
```
Question: {problem}
Answer: {solution}
```

**Why 8-shot in eval?** To match the official Qwen2.5-Math benchmark protocol (8-shot GSM8K, 4-shot MATH), enabling direct comparison of fine-tuned model scores against published numbers. Without this, results are not comparable to the literature.

**The measurement caveat**: The 8-shot context influences the model's output style via in-context learning. `avg_tokens_easy` measured under 8-shot conditions may not equal what the model produces in 0-shot deployment. However, since baseline and budget models are evaluated under identical 8-shot conditions, the *relative* improvement (DPO vs baseline) is valid — the absolute numbers just reflect benchmark conditions, not production deployment.

**In production** (real deployment), you would use 0-shot. Running a 0-shot ablation alongside the 8-shot eval is the correct way to measure actual deployment token efficiency.

---

## Final Recommendation

1. Continue with `Qwen2.5-Math-1.5B` (base) as the DPO starting point.
2. Do not switch to the Instruct variant.
3. SFT cold-start is not worth adding at this stage — training is already converging.
4. To measure real deployment efficiency, add a 0-shot eval pass alongside the standard 8-shot benchmark eval.
