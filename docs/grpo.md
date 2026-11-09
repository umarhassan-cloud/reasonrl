# GRPO in reasonrl

Implementation: [`src/reasonrl/rl/grpo.py`](../src/reasonrl/rl/grpo.py), [`advantages.py`](../src/reasonrl/rl/advantages.py), [`rollout.py`](../src/reasonrl/rl/rollout.py), [`buffer.py`](../src/reasonrl/rl/buffer.py).

## One training step

```
prompts q_1..q_N  ──repeat G──▶  N·G prompts (group-major)
        │
        ▼  sample_completions (own KV-cache loop, temperature/top-p)
completions o_{n,1..G}, action_mask, truncation flags
        │
        ▼  RewardFunction.score_batch  (verifier + format + hacking guards, threads for code)
rewards r_{n,i}
        │
        ▼  group_relative_advantages   A_{n,i} = (r_{n,i} − mean_i r_n) / (std_i r_n + ε)
        │
        ▼  no-grad forwards: old logprobs (policy), ref logprobs (frozen SFT model, if β>0)
RolloutBatch ──▶ RolloutBuffer
        │
        ▼  for μ epochs × minibatches:   loss = grpo_loss(...);  AdamW step
```

## Objective

For prompt `q` and its `G` samples:

```
J(θ) = E[ 1/G Σ_i 1/|o_i| Σ_t  min(ρ_{i,t} A_i, clip(ρ_{i,t}, 1−ε, 1+ε_high) A_i)  −  β · KL_k3(π_θ ‖ π_ref)_{i,t} ]
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})
KL_k3   = exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1
```

This is the DeepSeekMath formulation (Shao et al., 2024), also used for DeepSeek-R1. Properties worth knowing:

* **No critic.** The group mean is the baseline. Memory is policy + reference only, versus four models for PPO with a reward model and a value head.
* **The KL goes in the loss, not the reward.** PPO-RLHF subtracts `β·KL` from the reward. GRPO adds the k3 estimator directly to the per-token loss. k3 is unbiased and never negative.
* **μ = 1 is on-policy.** With one epoch and one minibatch, `ρ ≡ 1` at update time and the clip never activates. The gradient then reduces to REINFORCE with a group baseline. `test_grpo_loss_on_policy_gradient_matches_reinforce` checks this in closed form.
* **Zero-variance groups.** If all `G` samples get the same reward, the advantage is 0 and the group only contributes KL gradient. `groups/zero_std_frac` is logged every step. `filter_zero_std_groups` drops these groups, as DAPO's dynamic sampling does.

## Variants (all config switches, all tested)

| knob | values | reference |
|---|---|---|
| `grpo.adv_mode` | `std` · `mean_only` · `rloo` | GRPO · Dr. GRPO (Liu et al., 2025) · RLOO (Ahmadian et al., 2024) |
| `grpo.loss_agg` | `seq_mean` · `token_mean` · `constant` | DeepSeekMath · DAPO (Yu et al., 2025) · Dr. GRPO |
| `grpo.clip_eps_high` | `null` or e.g. `0.28` | DAPO "clip-higher" |
| `grpo.kl_coef` | `0.04` → `0` | `0` removes the reference model entirely |
| `grpo.kl_estimator` | `k1` · `k2` · `k3` | Schulman (2020), "Approximating KL divergence" |
| `grpo.ppo_epochs`, `minibatch_size` | μ, minibatch | μ > 1 reuses samples and makes the clip active |

`seq_mean` normalizes each sequence by its own length, which gives long wrong answers a smaller per-token penalty. This is the length bias that Dr. GRPO identifies. When length growth matters, use `token_mean` or `constant`.

## Correctness details that are easy to get wrong

1. **Token alignment.** `token_logprobs` returns `[B, T−1]`, where entry `j` scores token `j+1`. `action_mask[:, j] = 1` iff token `j+1` was generated, including EOS. Prompt tokens and post-EOS padding are masked out.
2. **Left padding and positions.** Prompts are left-padded for batched sampling. Position ids come from `cumsum(attention_mask) − 1`, both during sampling and when recomputing. A test checks that a padded row gives the same log-probs as the same sequence run alone.
3. **KV-cache equivalence.** A test checks that greedy sampling with the cache produces the same tokens as full recomputation.
4. **No re-tokenization.** The loss is computed on the exact token ids that were sampled. Decoding to text and re-encoding can change token boundaries and silently corrupt `ρ`.
5. **Dropout off.** Rollouts and log-prob recomputation run with `model.eval()`, so `π_old` and `π_θ` see the same network. `update()` switches to `train()` only so that gradient checkpointing works. Tiny models are built with zero dropout.
6. **fp32 log-softmax.** Logits are upcast before `logsumexp` to avoid bf16 error in `ρ`.

## Logged metrics (per step, `metrics.jsonl`)

`reward/{mean,std,correct,format,format_valid_rate,guarded_frac,pass_at_group}`, `groups/zero_std_frac`, `completion/{len_mean,truncated_frac}`, `hack/<flag>` for every detector, `train/{loss,pg_loss,kl,clip_frac,ratio_mean,grad_norm,lr}`, `time/{rollout_s,update_s}`, and periodic `eval/*`. A sample of completions per step goes to `samples.jsonl` so hacks can be audited by reading them.

## Scaling notes

* Rollouts dominate wall-clock: on CPU about 85% of each step was spent in rollout. On GPU, swap `sample_completions` for vLLM and sync weights every step. The `RolloutBatch` interface does not change.
* With `kl_coef=0` no reference forward runs, which saves about one third of the step FLOPs and all reference memory.
* LoRA (`lora.enabled`) cuts optimizer state to the adapters. Checkpoints are saved merged.
