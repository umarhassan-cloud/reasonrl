# reasonrl: Design

This document records how the system is built and why. Algorithm details are in [`docs/grpo.md`](docs/grpo.md), the reward and sandbox design in [`docs/rewards.md`](docs/rewards.md), and the decontamination method in [`docs/contamination.md`](docs/contamination.md).

## 1. Goals and non-goals

**Goals**
1. A faithful, readable implementation of SFT → GRPO and SFT → DPO for 0.5B–3B causal LMs in plain PyTorch and transformers, with no TRL/OpenRLHF dependency. Every line of the loss and the rollout should be auditable.
2. Experimental hygiene that holds up in review: typed configs, seeds, recorded environment and git SHA, a shared eval protocol, paired statistics, decontaminated eval, and reward-hacking audits.
3. CPU-testable end to end: every code path runs on a tiny random model in CI without network access.

**Non-goals**
* Peak throughput. There is no vLLM rollout engine and no FSDP/DeepSpeed sharding. §6 lists the seams where these plug in.
* A hardened multi-tenant code sandbox. The sandbox is defense in depth for research use, and §5 explains the threat model.

## 2. Package layout and dependencies

```
config.py ──────────────────────────────────────────────┐  typed dataclasses, YAML, base:, overrides
seed.py  logging_utils.py                               │
data/      dataset.py (schema, templates, datasets)     │
           collators.py (padding)                       │
models/    load.py (HF / tiny-random, checkpoints)      │
           lora.py (native LoRA + PEFT backend)         │
utils/     math_parse.py  sandbox.py  text.py           │  pure Python, no torch
rewards/   math / code / format / hacking / composite   │  depends on utils + data.Example
sft/       trainer.py (optimizer/scheduler shared)      │
rl/        rollout.py (sampling, logprobs)              │
           advantages.py buffer.py grpo.py dpo.py       │
eval/      metrics.py contamination.py benchmarks.py    │
pipeline.py  ◀── stage runners used by scripts/* and the ablation harness
```

Rewards and parsers do not import torch, so they can run in a separate reward-worker process if needed. Trainers receive dependencies through their constructors (policy, reference, tokenizer, reward function, logger), so tests can inject a tiny model and a stub reward.

## 3. Key decisions

| decision | alternative | why |
|---|---|---|
| Custom sampling loop with KV cache | `model.generate` | The loss must use the exact sampled token ids, masks and positions. `generate` adds logits processors and has version-specific left-padding behaviour. Our loop is tested to match full recomputation. |
| Prompt and completion tokenized separately | tokenize the joined string | Joining can merge tokens at the boundary, which would make the SFT/DPO token sequence differ from what the policy samples. |
| One shared prompt template for SFT, RL, DPO and eval | per-stage templates | A template mismatch between stages is a common silent cause of regressions. |
| Frozen deep copy as reference | LoRA "disable adapter" reference | Simpler and works for full fine-tuning. With LoRA, the adapter-off trick would save one model copy, and the seam is `make_reference_model`. |
| Reference log-probs precomputed once for DPO | recompute every step | The reference never changes, so this saves one forward pass per step. |
| Strict answer extraction by default | last-number fallback | The fallback was the root cause of the prompt-copy hack (PAPER §6). |
| Binary code reward | partial credit | Partial credit rewards special-casing the visible-looking tests. |
| Reward = 0 on guarded hack, not a negative reward | negative penalty | A negative value would push the advantage of hacked samples below honest failures. That is fine, but it also makes false positives costly. The penalty is configurable through `reward.hacking_penalty`. |
| Unknown config keys raise | ignore unknown keys | A typo in an ablation override would otherwise silently run the baseline. |
| Checkpoints saved merged | save adapters only | Every downstream stage loads a plain HF model. |

## 4. Fair GRPO-vs-DPO protocol

The two methods differ in exactly one respect: **online versus offline use of the same verifier**.

| controlled variable | GRPO | DPO |
|---|---|---|
| initialization | `runs/sft/final` | `runs/sft/final` |
| training prompts | `data.train_files` | `data.train_files` |
| reward / labels | `RewardFunction` (same config) | the same `RewardFunction` labels the pairs |
| samples per prompt | `group_size = 8` | `samples_per_prompt = 8` |
| sampling temperature | 1.0 | 1.0 |
| reference / anchor | KL to SFT, β = 0.04 | implicit KL to SFT, β = 0.1 |
| off-policy data | none | none (`gold_as_chosen=false`) |
| eval | `configs/eval.yaml`, clean subset, greedy, same seed | same |

GRPO sees `steps × prompts_per_step × G` samples in total, while DPO sees `|train| × samples_per_prompt`. Results tables report both counts. Comparisons are made either at equal sample counts or with the difference stated explicitly. Significance is tested with a paired bootstrap over eval items (`eval/metrics.py`).

## 5. Sandbox threat model

**In scope:** a policy that, under RL pressure, finds that touching the harness, the filesystem or the comparison operator earns reward. This is a real failure mode. Examples: `AlwaysEqual` objects, printing fake verdicts, exiting early, hard-coding outputs. All of these are blocked, and each has a test.

**Out of scope:** a deliberately adversarial human attacker. CPython in-process restrictions cannot be made airtight. For untrusted large-scale RL, run `run_tests` inside a container or gVisor sandbox with no network. The function signature is the isolation boundary, so swapping the backend does not touch the reward code.

## 6. Scaling seams

| concern | current | scale-up path |
|---|---|---|
| rollout throughput | HF forward + KV cache | vLLM / SGLang engine with weight sync each step. Replace `sample_completions`. |
| model memory | single device, optional LoRA + gradient checkpointing | `accelerate` / FSDP wrapping in the `pipeline.run_*` functions |
| reward latency | thread pool of sandbox subprocesses (`reward.code_workers`) | remote reward workers. Rewards are pure functions of `(Example, str)`. |
| reference model | full copy | LoRA adapter-off reference; or `kl_coef=0` (DAPO) |
| logging | JSONL + stdout | add a W&B sink in `MetricsLogger.log` |

## 7. Reproducibility checklist

* `seed` drives Python, NumPy and torch, plus a separate data-order RNG (`random.Random(seed)`). Evaluation re-seeds before **every** checkpoint, so all checkpoints see identical sampling noise.
* `deterministic: true` turns on deterministic torch kernels.
* Every run directory contains `config.json` (fully resolved), `env.json` (versions, GPU, git SHA) and `metrics.jsonl`. Every checkpoint contains `experiment.yaml`.
* The sample data is generated deterministically by `scripts/prepare_data.py synthetic --seed 1234`.

## 8. Testing strategy

| layer | test | technique |
|---|---|---|
| parsers / verifiers | `test_math_reward.py`, `test_code_reward.py` | table-driven; adversarial inputs (SymPy injection, sandbox escapes) |
| RL math | `test_grpo_advantages.py`, `test_dpo_loss.py` | closed-form gradients (REINFORCE equivalence at ρ = 1, clip zeroing, DPO log 2 at init) |
| numerics | `test_grpo_advantages.py` | KV cache vs recompute; left padding vs unpadded log-probs |
| trainers | `test_sft_step.py`, GRPO/DPO trainer tests | tiny model: loss decreases, parameters change, the reference stays frozen, checkpoint round-trip |
| hacking | `test_hacking_detection.py` | each hack reproduced, detected, and given zero reward |
| contamination | `test_contamination.py` | planted leaks in the sample data equal the flagged set |
