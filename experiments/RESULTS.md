# Results

This file separates **measured** results (produced by code in this repo, on CPU, with logs committed under `experiments/curves/`) from **projected** results (what we expect at 0.5B scale on a GPU; *not run*). Read the provenance column before reading any number.

| provenance | meaning |
|---|---|
| **measured** | Produced by the commands listed, on CPU, with a 0.54M-parameter random-init char-level GPT-2. Raw logs are in `experiments/curves/cpu_*`. |
| **projected** | Not run. Expected outcomes for Qwen2.5-0.5B-Instruct-class models on GSM8K/MBPP-style data, based on trends in the GRPO, DAPO, Dr. GRPO and Tülu 3 reports. Each one is a hypothesis to test, not a claim. |

---

## 1. Measured: CPU pipeline runs

### 1.1 Sample data (disjoint train/eval templates; clean subset n = 30 of 35)

```bash
python scripts/train_sft.py  --config configs/smoke/sft_tiny.yaml
python scripts/train_grpo.py --config configs/smoke/grpo_tiny.yaml
python scripts/train_dpo.py  --config configs/smoke/dpo_tiny.yaml
python scripts/evaluate.py   --config configs/smoke/eval_tiny.yaml
```

| run | pass@1 (greedy) | format valid | hack-flag rate | Δ vs SFT [95% CI] |
|---|---|---|---|---|
| SFT | 0.000 | 0.067 | 0.000 | – |
| SFT + GRPO (30 steps, 4×G4) | 0.000 | 0.000 | 0.067 | +0.000 [+0.000, +0.000] |
| SFT + DPO (140 pairs) | 0.000 | 0.033 | 0.000 | +0.000 [+0.000, +0.000] |

Contamination: 5/35 eval items were flagged and excluded. These are exactly the 5 planted leaks (§3).

Training signals (`cpu_smoke_*`):

* **SFT:** train loss 4.70 → 0.25 in 150 steps. Eval loss is 3.30 at step 100. The model overfits 70 examples and does not transfer to unseen templates, which is the expected result.
* **GRPO (mean of steps 1–10 → 21–30):** reward 0.024 → 0.046, format reward 0.19 → 0.28, KL to SFT 0.033 → 0.120, zero-variance groups 0.38 → 0.25. Rollout took 85% of wall-clock.
* **DPO:** 140 pairs. The SFT model was correct on only 0.7% of samples, so this run used `gold_as_chosen`. Loss 0.668 → 0.297, implicit-reward accuracy 0.50 → 1.00, margin 0.07 → 1.18, all on training pairs.

**Reading.** The pipeline is mechanically sound: losses fall, the KL grows from 0, format reward rises, and every metric is populated. On held-out problems the tiny model learns nothing, which is the correct result for a model this size.

### 1.2 Toy addition (in-distribution; eval facts disjoint from train facts; n = 30)

```bash
python scripts/train_sft.py  --config configs/smoke/toy_sft.yaml   # SFT under-trained on purpose (120 steps)
python scripts/train_grpo.py --config configs/smoke/toy_grpo.yaml  # 40 steps, 8 prompts x G=8
python scripts/train_dpo.py  --config configs/smoke/toy_dpo.yaml   # on-policy pairs, no gold
python scripts/evaluate.py   --config configs/smoke/toy_eval.yaml
```

| run | pass@1 (greedy) | 95% CI | format valid | hack-flag rate | Δ vs SFT [95% CI], p |
|---|---|---|---|---|---|
| SFT | 0.067 | [0.000, 0.167] | 1.000 | 0.933 | – |
| SFT + GRPO | 0.033 | [0.000, 0.100] | 1.000 | 1.000 | −0.033 [−0.167, +0.067], p = 0.79 |
| SFT + DPO (60 pairs / 30 prompts) | 0.167 | [0.067, 0.300] | 1.000 | 0.967 | +0.100 [−0.067, +0.267], p = 0.33 |

**Reading.** None of these differences is significant. Greedy outputs are flagged `constant_answer` on 93–100% of eval items: the under-trained SFT policy gives roughly the same sum to every question, so its 2/30 "correct" answers are chance. GRPO training reward rose from 0.148 to 0.162 (steps 21–30) and then fell back to 0.147, with train correctness flat at 5–7% (≈ chance for 19 possible sums). At this scale 40 GRPO steps cannot teach addition facts the policy never acquired. RL sharpens existing ability; it does not create it from nothing, which matches the DeepSeek-R1 and Dr. GRPO observations. The DPO gain is suggestive but is a single seed with p = 0.33.

These runs are **pipeline validation, not evidence about GRPO vs DPO.** They produced the bug findings in §2, which are the real output of the CPU work.

---

## 2. Reward-hacking and diagnostics findings

### Found in measured runs (and fixed)

| id | what we saw | root cause | fix | regression test |
|---|---|---|---|---|
| **F1** | `hack/empty_answer` ≈ 24% during the sample-data GRPO run; 13.3% of SFT eval outputs flagged | Detector fired on *unterminated* tags (`<answer>424.</ank>`), i.e. truncated or malformed output, not hacking | Count only complete `<answer>…</answer>` tags. After the fix, the SFT eval flag rate is 0.0%. | `test_unterminated_answer_is_not_a_hack` |
| **F2** | `reward/guarded_frac` ≈ 40% in toy GRPO, **zeroing correct samples** | The batch-level `constant_answer` rule counted an answer appearing in *any* of a prompt's G samples. At G = 8, T = 1 an uncertain honest policy trips it. | Use each prompt's **modal** answer (≥ 50% of its group). Guarded fraction fell from about 40% to under 1%, and mean reward rose from 0.086 to 0.15 at similar train correctness (4.9% vs 5.8%). | `test_constant_answer_uses_modal_answer_per_group` |
| **F3** | Toy GRPO `completion/truncated_frac = 1.0`, `groups/zero_std_frac = 1.0`, KL = 0 | `max_new_tokens = 40` was shorter than the 45-token target. Every sample was truncated, so every group had identical zero reward and there was no gradient. | Raise the budget to 64. Treat `zero_std_frac ≈ 1` as an alarm. | config fix; the metric is now documented in `docs/grpo.md` |
| **F4** | SFT policy answers the same sum for most prompts and is sometimes right by luck | Mode collapse in an under-trained policy: a "constant answer" policy that is not adversarial | `constant_answer` guard denies reward for lucky hits; eval reports `hack_flag_rate` next to pass@1 | `test_constant_answer_across_prompts` |

F2 matters most. A guard that is too aggressive does not just add noise: it removes reward from **correct** answers. The rule is to validate every hacking detector on an honest policy before turning it on as a guard.

### Provoked by construction (unit tests reproduce each hack, then check it earns 0)

| hack | vulnerable configuration | fix |
|---|---|---|
| `<think></think><answer></answer>` earns full format reward | `format_mode: naive` | strict format (non-empty, single tag, ordered, no trailing text) plus `empty_answer` guard |
| copy the problem; the last number in the text is "the answer" | `strict_extraction: false` | strict extraction (tag or `\boxed` only) plus `prompt_copy` guard |
| several boxed answers ("shotgun") | first-match extraction | last-tag extraction plus `multiple_answers` guard |
| always answer "42" / "0" | per-sample-only checks | batch-level `constant_answer` (modal, F2-corrected) |
| `class AlwaysEqual: __eq__ → True` | `assert out == expected` | type-strict comparison in the sandbox |
| print a fake verdict JSON | parsing the child's stdout | nonce-prefixed verdict line; user stdout captured |
| `raise SystemExit(0)` before tests run | exit code as the signal | runner catches `BaseException`; `SystemExit` is not exposed |
| `return {(1, 2): 3, ...}[(a, b)]` | visible tests | hidden tests plus `hardcoded_outputs` guard |
| `().__class__.__bases__[0].__subclasses__()` | in-process exec | AST screen (dunder and frame attributes), isolated interpreter |

Run them yourself with `pytest tests/test_hacking_detection.py tests/test_code_reward.py -v`.

---

## 3. Contamination (measured)

`python scripts/check_contamination.py --train data/sample/*_train.jsonl --eval data/sample/math_eval.jsonl data/sample/code_eval.jsonl --mask-numbers`

| setting | flagged | rate | planted leaks caught | false positives |
|---|---|---|---|---|
| 8-gram, raw | 5 / 35 | 14.3% | 5 / 5 (templated leak only barely: 0.60 vs threshold 0.5) | 0 |
| 8-gram, numbers masked | 5 / 35 | 14.3% | 5 / 5 (all at 1.00) | 0 |

---

## 4. Projected: GPU-scale experiments (NOT RUN)

Setup: Qwen2.5-0.5B-Instruct; SFT on GSM8K + MBPP train traces; GRPO 200 steps × 64 samples; DPO on 8 samples/prompt from SFT; eval on decontaminated GSM8K test + MBPP test, greedy, 3 seeds. Run with `configs/{sft,grpo,dpo,eval}.yaml` after `prepare_data.py hf`. The curves in `experiments/curves/illustrative_*` encode these expectations.

| run | GSM8K pass@1 | MBPP pass@1 | format valid | hack-flag rate | mean length | basis |
|---|---|---|---|---|---|---|
| SFT | ≈ 0.31 | ≈ 0.28 | ≈ 0.86 | low | ≈ 180 tok | starting point |
| SFT + GRPO (β = 0.04, G = 8) | **≈ 0.44** | **≈ 0.33** | ≈ 0.98 | ≈ 1% | ≈ 235 tok | RLVR gains on GSM8K-type data in DeepSeekMath and Tülu 3 |
| SFT + DPO (β = 0.1) | ≈ 0.37 | ≈ 0.30 | ≈ 0.95 | ≈ 1% | ≈ 190 tok | offline pairs; less exploration than online RL |
| GRPO, no KL | ≈ 0.42, then drifting down | ≈ 0.32 | ≈ 0.95 | ≈ 6% | grows steadily | faster reward climb; DAPO removes KL but adds other controls |
| GRPO, no KL, no guards | ≈ 0.39 | – | ≈ 0.96 | ≈ 12% | grows fastest | train reward highest, held-out lowest: the hacking signature |
| GRPO, G = 4 (64-sample budget) | ≈ 0.425 | – | – | – | – | more zero-variance groups (≈ 0.5 vs 0.35) |
| GRPO, correctness only (w_f = 0) | ≈ 0.43 | – | ≈ 0.93 | – | – | format drifts slightly without shaping |
| GRPO, naive format, unguarded | ≪ SFT | – | ≈ 1.0 (!) | high (`empty_answer`) | collapses | expected format-only collapse |

**What would falsify our design choices:**
* GRPO with KL does not beat DPO at equal samples. Then online exploration is not paying for itself at 0.5B.
* No-KL ≥ KL on held-out accuracy across 3 seeds. Then the reference model is not worth its memory, as DAPO argues.
* Removing guards does not change held-out accuracy. Then the guards are either unnecessary or causing F2-style false positives.

## 5. Reproducing

```bash
pip install -e ".[dev]"
python scripts/prepare_data.py synthetic          # regenerates data/sample deterministically
pytest -q                                         # all CPU tests (≈ 1 min)
# measured CPU runs: see the commands in §1
python scripts/run_ablations.py experiments/ablations/*.yaml --dry-run
```
