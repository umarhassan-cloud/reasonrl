# Ablations

Each YAML file is a base config (`base: ../../configs/grpo.yaml`) plus `variants:`, where each variant maps a name to dotted overrides. `scripts/run_ablations.py` trains each variant × seed **from the same SFT checkpoint**, evaluates it under the shared protocol (`configs/eval.yaml` settings, clean subset only), and writes `results/<ablation>.{json,md}` with mean ± std over seeds.

```bash
# resolve and print every variant without training (also runs in CI)
python scripts/run_ablations.py experiments/ablations/*.yaml --dry-run

# full run, 3 seeds (GPU)
python scripts/run_ablations.py experiments/ablations/remove_kl.yaml --seeds 0 1 2

# CPU smoke of the harness
python scripts/run_ablations.py experiments/ablations/group_size.yaml --seeds 0 --only g4_budget32 \
  --out runs/abl_smoke \
  -o model.name_or_path=tiny-random model.dtype=float32 model.gradient_checkpointing=false device=cpu \
     grpo.init_from=runs/smoke/sft grpo.steps=3 grpo.max_new_tokens=48 grpo.prompts_per_step=2 \
     grpo.eval_every=0 eval.max_new_tokens=48 data.max_prompt_len=512 output_dir=runs/abl_smoke
```

| file | question | variants |
|---|---|---|
| `remove_kl.yaml` | Is the KL anchor to the SFT model needed? | β ∈ {0.04, 0.01, 0}; β = 0 without guards |
| `group_size.yaml` | G = 4 vs G = 8 at a **fixed 64-sample budget**, and G = 4 at half the budget | `g4_budget64`, `g8_budget64`, `g4_budget32` |
| `reward_shaping.yaml` | Does format shaping help, and which reward designs can be hacked? | correctness only, + format, format-gated, naive format (unguarded), lenient extraction (unguarded) |

## Controls

* Same SFT init, data order seed, eval set, decoding and token budget for every variant.
* A group-size comparison only makes sense at a fixed sample budget. Otherwise G = 8 simply gets twice the compute. `g4_budget32` separates the effect of G from the effect of compute.
* Report ≥ 3 seeds. Single-seed RL differences under about 2 points on a few hundred eval items are usually noise. The eval uses a paired bootstrap for this reason.
* Unguarded variants exist to **provoke** hacks. Their train reward is not comparable to the others, so compare held-out correctness only.

Results and interpretation: [`../RESULTS.md`](../RESULTS.md).
