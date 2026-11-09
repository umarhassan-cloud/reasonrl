# reasonrl: Online GRPO vs. Offline DPO with Verifiable Rewards, Hacking Guards, and Decontaminated Evaluation for Small Language Models

*Technical report, v0.1.* Every number in this report is labelled **measured** (CPU runs in this repository, logs committed) or **projected** (expected at 0.5B scale, not run). We do not mix the two.

---

## Abstract

We present reasonrl, a compact open implementation of the post-training recipe behind recent reasoning models: supervised fine-tuning (SFT), followed by reinforcement learning from verifiable rewards (RLVR) with Group Relative Policy Optimization (GRPO; Shao et al., 2024; DeepSeek-AI, 2025), compared with Direct Preference Optimization (DPO; Rafailov et al., 2023) trained on preference pairs labelled by **the same verifier**. The system has four parts: (i) math verifiers (strict extraction, exact-rational and guarded-SymPy equivalence) and code verifiers (hidden unit tests in a layered sandbox); (ii) a catalogue of nine reward-hacking detectors, each reproduced by a test that also checks the fix; (iii) n-gram decontamination with number masking for templated leaks; and (iv) a matched-budget GRPO-vs-DPO protocol with paired-bootstrap significance tests. On CPU, with a 0.54M-parameter model, we confirm the pipeline works end to end. As expected at that scale, we find **no significant differences between methods**. Validating the detectors on an honest policy exposed two of our own bugs. In the worse one, a batch-level "constant answer" guard zeroed about 40% of rewards, including correct ones. The report closes with projected GPU-scale results and the outcomes that would falsify our design choices.

## 1. Introduction

Recent reasoning models (DeepSeek-R1; Tülu 3's RLVR stage) show that a binary, programmatically checked reward is enough to improve multi-step reasoning, without a learned reward model. Three practical questions remain under-documented for *small* models and *small* budgets:

1. **Online vs. offline.** GRPO samples from the current policy at every step. DPO learns from a fixed set of pairs. If both use the same verifier and the same sampling budget, how much of GRPO's gain comes from being online?
2. **Reward integrity.** A verifiable reward is only as good as its extractor, comparator and sandbox. Which hacks appear, and do the guards against them cost anything?
3. **Measurement.** Small eval sets, templated data and train/test overlap can produce gains that are noise or memorization.

Our contributions are engineering and methodology rather than state-of-the-art numbers. They are: a readable reference implementation in plain PyTorch with no RL framework dependency; a reward-hacking catalogue with a reproducing test for each entry; a fair comparison protocol; and an honest report that includes negative results and our own detector bugs.

## 2. Background

**GRPO** (Shao et al., 2024) drops PPO's value network (Schulman et al., 2017). For each prompt q it samples G completions and uses the group-normalized reward as the advantage of every token:

```
A_i = (r_i − mean(r_1..G)) / std(r_1..G)
J(θ) = E[ 1/G Σ_i 1/|o_i| Σ_t min(ρ_{i,t} A_i, clip(ρ_{i,t}, 1±ε) A_i) − β KL_k3(π_θ ‖ π_ref) ]
```

The KL to the reference model is estimated per token with the unbiased k3 estimator and added to the loss, not to the reward. DeepSeek-R1 applies this objective to rule-based accuracy and format rewards. Follow-up work changes the details: DAPO (Yu et al., 2025) removes the KL, adds asymmetric "clip-higher" and dynamic sampling of zero-variance groups, and uses token-level aggregation. Dr. GRPO (Liu et al., 2025) argues that the std normalization and the per-sequence length normalization add difficulty and length biases. RLOO (Ahmadian et al., 2024) uses a leave-one-out baseline. All of these are switches in our implementation (Table 1).

**DPO** (Rafailov et al., 2023) optimizes the closed-form RLHF policy directly on pairs:

```
L = −log σ(β[(log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x))])
```

We also implement IPO (Azar et al., 2023), conservative DPO via label smoothing, and a hinge variant. Tülu 3 (Lambert et al., 2024) uses DPO for general preferences and RLVR for verifiable skills. We put both on the *same* verifiable signal.

**Table 1.** GRPO variants implemented (`configs/grpo.yaml`).

| axis | options |
|---|---|
| advantage | `std` (GRPO), `mean_only` (Dr. GRPO), `rloo` |
| aggregation | `seq_mean` (GRPO), `token_mean` (DAPO), `constant` (Dr. GRPO) |
| clipping | symmetric ε, clip-higher `ε_high` (DAPO) |
| KL | β ∈ [0, ∞), estimators k1/k2/k3; β = 0 drops the reference model |
| sampling | μ PPO epochs, minibatches, zero-variance group filtering |

## 3. Method

### 3.1 Pipeline

```
curated traces ─▶ SFT ─┬─▶ GRPO (online: sample G, verify, group-normalize, clipped PG + KL)
                       └─▶ sample 8/prompt ─▶ verify ─▶ pairs ─▶ DPO (offline)
                                 both ─▶ shared eval protocol on decontaminated held-out set
```

All stages share one prompt template and the response format `<think>…</think><answer>…</answer>`. We tokenize prompt and completion separately, so the token ids the policy samples are exactly the ids used in training. Sampling uses our own KV-cached loop. Tests check that it matches full recomputation and that left padding leaves the log-probs unchanged.

### 3.2 Rewards

`r = 1.0·correct + 0.1·format`. If any guarded hacking flag fires, `r = −penalty` (default 0).

* **Math.** Strict extraction takes the last `<answer>` tag or the last `\boxed{}`. The answer is normalized (LaTeX, units, thousands separators) and compared by string, exact rational, float tolerance, tuple, and then SymPy. SymPy's parser calls `eval`, so model output reaches it only after a character and identifier whitelist and size limits.
* **Code.** Hidden I/O tests with type-strict comparison, run in a separate `python -I -S` process. Defenses: AST screen (imports, dangerous builtins, dunder and frame attributes), restricted builtins, timeout and rlimits, and a nonce-authenticated verdict channel. The reward is binary (all tests pass).
* **Format.** The strict mode requires exactly one non-empty answer, a non-trivial think block, the correct order and no trailing text. The naive mode only checks that the tags exist, and is kept as a control we know is hackable.

### 3.3 Fair comparison protocol

GRPO and DPO start from the same SFT checkpoint and use the same train prompts, verifier configuration, sampling temperature and samples per prompt (G = 8 vs. 8). DPO gets no off-policy gold solutions. Evaluation uses the same decontaminated items, greedy decoding, token budget and per-checkpoint re-seeding. Evaluation scores **correctness only**, never the shaped reward. Differences are reported with 95% paired-bootstrap intervals over items.

### 3.4 Decontamination

For each eval item we compute the fraction of its word 8-grams found anywhere in the training prompts (8 follows Tülu 3; items shorter than 8 words fall back to an exact match). With number masking, every number becomes `<num>` so that templated near-duplicates match. Flagged items (≥ 0.5) are excluded from the reported eval.

## 4. Experimental setup

**Measured (CPU).** Policy: a randomly initialized 2-layer GPT-2 (d = 128, 0.54M parameters) with a character-level tokenizer. Two datasets: (a) *sample*: 60 math and 10 code training items; 35 eval items whose templates are disjoint from training, plus 5 planted leaks; (b) *toy addition*: the 100 single-digit sums split by fact into 70 train and 30 eval. Budgets: SFT 120–150 steps; GRPO 30–40 steps of 16–64 samples; DPO 1–2 epochs. One seed. Commands are in `experiments/RESULTS.md`.

**Projected (GPU, not run).** Qwen2.5-0.5B-Instruct; GSM8K + MBPP; GRPO 200 × 64 samples, lr 1e-6, β = 0.04; DPO β = 0.1, lr 5e-7; 3 seeds; one A100-class GPU. Configs are `configs/{sft,grpo,dpo,eval}.yaml`.

## 5. Results

### 5.1 Measured

**Table 2.** CPU runs, greedy pass@1 on the clean eval subset (n = 30). Δ is relative to SFT with a 95% paired-bootstrap CI.

| data | SFT | SFT + GRPO | SFT + DPO | note |
|---|---|---|---|---|
| sample (disjoint templates) | 0.000 | 0.000 (Δ 0.000) | 0.000 (Δ 0.000) | no transfer at 0.54M params |
| toy addition (disjoint facts) | 0.067 | 0.033 (Δ −0.033 [−0.167, +0.067]) | 0.167 (Δ +0.100 [−0.067, +0.267], p = 0.33) | outputs 93–100% `constant_answer` |

At this scale **we observe no significant differences between methods**, and the sub-chance-looking accuracies come from mode collapse. Greedy SFT gives almost the same sum for every question. On the toy task, GRPO training reward rose from 0.148 to 0.162 and returned to 0.147 over 40 steps, while train correctness stayed at 5–7%. This agrees with the view that RLVR amplifies abilities already present in the base policy and cannot create them from nothing within a small budget. The mechanics behave as expected: SFT loss falls (4.70 → 0.25), GRPO's KL to the reference grows from 0 (0.03 → 0.12) while format reward rises (0.19 → 0.28), and DPO's implicit-reward accuracy on training pairs goes from 0.50 to 1.00 with the loss falling from 0.668 to 0.297.

### 5.2 Projected (not run)

**Table 3.** Projected GSM8K pass@1 for a 0.5B model. These are hypotheses derived from published trends, not results.

| SFT | + GRPO | + DPO | GRPO β = 0 | GRPO β = 0, no guards |
|---|---|---|---|---|
| ≈ 0.31 | ≈ 0.44 | ≈ 0.37 | ≈ 0.42, drifting | ≈ 0.39 (hack rate ≈ 12%) |

We expect online GRPO to beat offline DPO at an equal sample budget, mainly because GRPO keeps re-sampling the prompts where the current policy sits at the margin. DPO's pairs go stale as the policy moves away from the SFT model.

## 6. Failure modes: reward hacking and our own bugs

### 6.1 Hacks provoked by construction

We built each hackable configuration and confirmed that a hand-written exploit collects reward. We then confirmed that the fix brings the reward to zero (`tests/test_hacking_detection.py`, `tests/test_code_reward.py`).

| # | exploit | vulnerable design | fix |
|---|---|---|---|
| H1 | `<think></think><answer></answer>` | tag-presence format reward | strict format plus `empty_answer` guard |
| H2 | restate the problem; the last number wins | "last number" extraction | strict extraction plus `prompt_copy` guard |
| H3 | `\boxed{40} \boxed{41} \boxed{42}` | first- or any-match extraction | last-tag extraction plus `multiple_answers` guard |
| H4 | always answer "42" | per-sample checks only | batch-level `constant_answer` guard |
| H5 | `AlwaysEqual` with `__eq__ → True` | `assert out == exp` | type-strict comparison |
| H6 | print a fake verdict JSON | parsing child stdout | captured stdout plus nonce verdict line |
| H7 | `raise SystemExit(0)` | exit status as the signal | `BaseException` caught; `SystemExit` not exposed |
| H8 | literal lookup table for the visible tests | visible tests | hidden tests plus `hardcoded_outputs` guard |
| H9 | `().__class__.__subclasses__()` escape | in-process `exec` | AST dunder/frame screen plus isolated interpreter |

### 6.2 Found in measured runs

**F1: a false positive on malformed output.** The `empty_answer` detector flagged 24% of training samples and 13% of SFT eval outputs. Reading `samples.jsonl` showed that these were *unterminated* tags (`<answer>424.</ank>`). That is a truncated or malformed completion, not a hack. Restricting the detector to complete tags brought the eval flag rate to 0%. Rewards did not change, because strict extraction already scored these 0, but the monitoring dashboard had been overstating hacking by an order of magnitude.

**F2: a guard that removed reward from correct answers.** The batch-level `constant_answer` rule counted an answer as shared if it appeared in *any* of a prompt's G samples. With G = 8 at temperature 1, an uncertain but honest policy puts common values (such as "9") in some sample of most prompts. The guard then zeroed about 40% of rewards, including correct ones, which directly reduces the gradient signal from correct answers. Switching to each prompt's *modal* answer (≥ 50% of its group) cut the guarded fraction from 40% to under 1% and raised mean reward from 0.086 to 0.15 at similar correctness. **Lesson:** a hacking guard is a classifier with a false-positive rate. Measure that rate on honest policies before letting it change the reward.

**F3: a silent no-signal run.** A 40-token generation budget was shorter than the 45-token target. Every sample was truncated, every group had identical zero reward, and the GRPO gradient was exactly zero (KL = 0), with no error raised. The logged `completion/truncated_frac = 1.0` and `groups/zero_std_frac = 1.0` diagnosed it immediately. We recommend alerting when `zero_std_frac` is close to 1.

**F4: lucky hits from mode collapse.** An under-trained SFT policy answers almost every question with the same value and is correct whenever the gold answer happens to match. This "constant answer" behaviour arose naturally, not adversarially. The F2-corrected guard denies it reward, and eval tables report `hack_flag_rate` next to pass@1 so that chance-level accuracy is not misread.

## 7. Ablations

The harness (`scripts/run_ablations.py`) runs each variant × seed from the same SFT checkpoint under the shared protocol.

| ablation | variants | controlled | hypothesis |
|---|---|---|---|
| KL (`remove_kl.yaml`) | β ∈ {0.04, 0.01, 0}; β = 0 without guards | init, data order, eval | β = 0 climbs train reward fastest but drifts (length, format, hacking). The guards recover most of the held-out gap. |
| group size (`group_size.yaml`) | G = 4 vs 8 at 64 samples/step; G = 4 at 32 | sample budget | G = 8 has fewer zero-variance groups and lower variance. G = 4 sees 2× more prompts. We expect a small G = 8 advantage on hard data. |
| reward shaping (`reward_shaping.yaml`) | correctness only; + format; format-gated; naive format (unguarded); lenient extraction (unguarded) | everything else | Format shaping helps format validity, with ≤ 1 point effect on accuracy. The unguarded variants reproduce H1/H2 at scale. |

On CPU the ablation harness runs end to end and the configs are resolved in CI with `--dry-run`. Measured GPU ablation numbers are future work, and illustrative curves are provided under `experiments/curves/illustrative_*`.

## 8. Limitations

* No GPU-scale measurement yet. The headline method comparison is projected.
* CPU results are single-seed with n = 30. CIs are wide by construction, and we report them instead of point estimates alone.
* The sandbox is defense in depth, not a security boundary. Large-scale untrusted execution needs container isolation.
* Lexical decontamination misses paraphrases. Semantic and membership-inference checks are out of scope.
* The samples-matched comparison does not match FLOPs. GRPO's online sampling costs more wall-clock per sample than DPO's single offline pass.

## 9. Conclusion

reasonrl implements SFT → GRPO and SFT → DPO on a shared verifiable reward, with the measurement practices that make the comparison meaningful: decontaminated held-out sets, paired significance tests, matched sample budgets, and hacking detectors that are themselves tested. The most transferable lesson from the CPU work is about the guards: detectors and guards need the same validation as the model. Two of our own detectors were wrong in ways only visible by reading samples from real runs, and one of them suppressed correct answers.

## References

* Ahmadian, A., et al. (2024). *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.* arXiv:2402.14740.
* Azar, M. G., et al. (2023). *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO). arXiv:2310.12036.
* Brown, T., et al. (2020). *Language Models are Few-Shot Learners* (13-gram decontamination). NeurIPS.
* Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code* (unbiased pass@k). arXiv:2107.03374.
* Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K). arXiv:2110.14168.
* DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948.
* Hu, E., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
* Lambert, N., et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* arXiv:2411.15124.
* Liu, Z., et al. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO). arXiv:2503.20783.
* Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
* Schulman, J. (2020). *Approximating KL Divergence.* Blog post.
* Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
* Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). arXiv:2402.03300.
* Yu, Q., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476.
