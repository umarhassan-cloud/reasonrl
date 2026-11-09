# Training curves

Every file in this directory states its provenance. Do not mix the two kinds.

| file | provenance | content |
|---|---|---|
| `cpu_smoke_sft.{jsonl,csv}` | **measured** | SFT on the tiny 2-layer char-level GPT-2 (CPU, `configs/smoke/sft_tiny.yaml`) |
| `cpu_smoke_grpo.{jsonl,csv}` | **measured** | GRPO from that SFT checkpoint (CPU, `configs/smoke/grpo_tiny.yaml`) |
| `cpu_smoke_dpo.{jsonl,csv}` | **measured** | DPO from that SFT checkpoint (CPU, `configs/smoke/dpo_tiny.yaml`) |
| `cpu_toy_{sft,grpo,dpo}.{jsonl,csv}` | **measured** | toy-addition runs (`configs/smoke/toy_*.yaml`), recorded after the F1/F2 detector fixes |
| `illustrative_grpo_kl0.04.{json,csv}` | **illustrative** | expected shape: GRPO with KL, 0.5B-class model |
| `illustrative_grpo_nokl.{json,csv}` | **illustrative** | expected shape: GRPO without KL |
| `illustrative_grpo_runs.json` | **illustrative** | all GRPO variants, including unguarded no-KL and G = 4 vs 8 |
| `illustrative_dpo.{json,csv}` | **illustrative** | expected DPO loss, reward accuracy and margins |

**Measured** files are the raw `metrics.jsonl` from the run, copied unchanged and converted to CSV. They come from a 0.54M-parameter random-init model. They show that the full pipeline runs, that the metrics are wired correctly, and that the format is learned. They say nothing about reasoning ability. `cpu_smoke_grpo` was recorded **before** the F1 fix (RESULTS.md §2), so its `hack/empty_answer` column (≈ 24%) is the false positive that led to that finding. It is left uncorrected on purpose.

**Illustrative** files come from [`make_illustrative_curves.py`](make_illustrative_curves.py), which uses parametric shapes plus seeded noise. They encode the qualitative behaviour we expect at GPU scale and that the GRPO/DAPO/Dr. GRPO papers report: reward saturates; without KL, length and drift grow; the no-KL, unguarded run climbs fastest on train reward while held-out accuracy falls behind. They exist so plotting and reporting code can be built before GPU runs exist. **They are not results.** Replace them with measured runs (`runs/<name>/grpo/metrics.jsonl`) as soon as you have them.

## Plotting

```python
import json, matplotlib.pyplot as plt
d = json.load(open("experiments/curves/illustrative_grpo_runs.json"))
for name, s in d["runs"].items():
    plt.plot(d["steps"], s["eval/pass@1"], label=name)
plt.legend(); plt.title("ILLUSTRATIVE - not measured"); plt.show()
```
