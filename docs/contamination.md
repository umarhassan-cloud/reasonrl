# Contamination checks

Code: [`src/reasonrl/eval/contamination.py`](../src/reasonrl/eval/contamination.py), CLI: [`scripts/check_contamination.py`](../scripts/check_contamination.py).

## Why it matters here

RL with verifiable rewards improves held-out accuracy by sharpening behaviour the model already has. If an eval item, or a template-level near-copy of it, appears in the SFT or RL prompts, the gain partly measures memorization. Small post-training studies are especially exposed, because synthetic or templated math data produces many near-duplicates.

## Method

1. Normalize the text: lowercase, strip punctuation, collapse whitespace.
2. Optionally **mask numbers** (`--mask-numbers`). Every number becomes `<num>`, so `"Tom has 3 apples"` and `"Tom has 5 apples"` match. This catches templated leaks.
3. Build an inverted index from word n-grams to train ids, over all training prompts (SFT and RL).
4. For each eval item, compute:
   * `overlap`: the fraction of its n-grams that appear anywhere in the training set (containment, not Jaccard, so long train items cannot dilute it)
   * `best_train_id` and `best_pair_overlap`: the single most similar train item, for auditing
5. Flag the item if `overlap ≥ threshold` (default 0.5).
6. Items shorter than `n` tokens fall back to `n = len(item)`, which is effectively an exact-match test. Without this, short problems such as `"What is 7 + 3 * 4?"` could never be flagged.

The default is `n = 8`, following Tülu 3's decontamination. GPT-3 used 13-grams on much longer documents, and n = 13 misses most leaks in short math problems.

## Protocol

* `eval.use_clean_subset: true` (the default) evaluates **only** unflagged items. The report, including flagged ids and the rate, is stored in `results.json`.
* `--fail-above RATE` makes CI fail if a new eval set is too contaminated. CI runs this on the sample data.
* Report numbers on both the clean subset and the full set when they differ. A large gap is itself evidence of memorization.

## Sample data sanity check

`data/sample` contains **planted** leaks, marked `meta.planted`, so the checker has something real to find:

| eval id | planted as | overlap (raw) | overlap (masked) |
|---|---|---|---|
| `math-eval-x0..x2` | verbatim copies of train items | 1.00 | 1.00 |
| `math-eval-x3` | same template as `t_speed`, new numbers | 0.60 | 1.00 |
| `code-eval-x0` | verbatim copy of `count_vowels` | 1.00 | 1.00 |

All 5 are flagged and no clean item is (5/35 = 14.3%). `tests/test_contamination.py` asserts that the flagged set equals the planted set. Raw n-grams catch the template leak only narrowly (0.60 against a 0.5 threshold), which shows why masking matters.

## Limitations

Lexical overlap misses paraphrases and translations. For public benchmarks, also check (a) an embedding-similarity pass, if an embedding model is allowed, and (b) membership-inference signals such as min-k% log-prob on the base model. Both are out of scope for this embedding-free checker.
