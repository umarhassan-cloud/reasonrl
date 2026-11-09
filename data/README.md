# Data

## Schema (JSONL, one example per line)

| field | type | used by |
|---|---|---|
| `id` | str, unique across all files | everything |
| `task` | `"math"` or `"code"` | reward routing |
| `question` | str | prompt, contamination checks |
| `answer` | str | math verifier (gold final answer) |
| `solution` | str in `<think>…</think>\n<answer>…</answer>` format | SFT target, optional DPO `gold_as_chosen` |
| `tests` | list of `{"input": [...], "output": ...}` or assert strings | code verifier (hidden from the prompt) |
| `entry_point` | str | code verifier (function under test) |
| `source`, `meta` | provenance | auditing; `meta.planted` marks deliberate leaks |

## `sample/`: CPU demo data (deterministic, regenerate with `python scripts/prepare_data.py synthetic`)

| file | n | notes |
|---|---|---|
| `math_train.jsonl` | 60 | 10 arithmetic/word-problem templates × 6 |
| `math_eval.jsonl` | 28 | 6 **disjoint** templates × 4, plus 4 planted leaks (3 verbatim, 1 templated) |
| `code_train.jsonl` | 10 | hand-written function tasks with I/O tests |
| `code_eval.jsonl` | 7 | 6 disjoint tasks, plus 1 planted verbatim leak |

The planted leaks give `scripts/check_contamination.py` something to find. The clean eval subset excludes them automatically. The sample data only exercises the pipeline end to end. It is far too small to support claims about model quality.

## Real data

```bash
pip install datasets
python scripts/prepare_data.py hf --dataset gsm8k --out data/hf      # 7.47k train / 1.32k test
python scripts/prepare_data.py hf --dataset math  --out data/hf      # MATH (algebra subset)
python scripts/prepare_data.py hf --dataset mbpp  --out data/hf      # MBPP sanitized
python scripts/check_contamination.py --train data/hf/*_train.jsonl --eval data/hf/*_eval.jsonl \
    --mask-numbers --write-clean data/hf/clean_eval.jsonl
```

Then point `data.train_files` and `data.eval_files` in the configs at the new files. Licenses: GSM8K (MIT), MATH (MIT), MBPP (CC-BY-4.0). Check them before redistributing derived data.
