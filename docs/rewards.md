# Verifiable rewards and reward-hacking guards

Code: [`src/reasonrl/rewards/`](../src/reasonrl/rewards), [`utils/math_parse.py`](../src/reasonrl/utils/math_parse.py), [`utils/sandbox.py`](../src/reasonrl/utils/sandbox.py).

## Composite reward

```
r = w_c · correct  +  w_f · format          if no guarded hacking flag fires
r = −hacking_penalty                         otherwise (default 0)
```

Defaults are `w_c = 1.0` and `w_f = 0.1`. Correctness is binary and checked by a verifier. Format is a small shaping term, so a well-formatted wrong answer (0.1) is never worth more than a correct one (≥ 1.0). If `require_format_for_correctness` is set, correctness only counts when the format is fully valid.

**Evaluation never uses the shaped reward.** Held-out numbers are correctness only.

## Response format

```
<think> ... reasoning ... </think>
<answer> final answer </answer>          (math; \boxed{...} inside or outside is also accepted)
<answer> ```python ... ``` </answer>     (code)
```

## Math correctness

1. **Extraction** (`extract_final_answer`, strict by default). Take the last `<answer>` tag and unwrap a `\boxed{}` inside it. Otherwise take the last `\boxed{}` (with brace matching, so nested `\frac{}{}` works). Otherwise there is no answer. The lenient fallback ("answer is X", `#### X`, last number) exists only for ablations because it can be hacked.
2. **Normalization.** Strips `$`, `\text{}`, `\left`/`\right`, units, thousands separators, `x =`, trailing periods and `\dfrac`, and maps `\frac12` to `\frac{1}{2}`.
3. **Equivalence**, cheapest check first:
   - normalized string equality
   - exact rational equality (`3/4 == \frac{6}{8} == 0.75`)
   - relative float tolerance (`math_tolerance`)
   - elementwise comparison for tuples
   - **guarded SymPy** (`2\sqrt{2} == \sqrt{8}`)

**SymPy safety.** `sympy.parse_expr` calls `eval`, so the model's answer is converted to a SymPy string and only parsed if it passes all of these: a character whitelist, an identifier whitelist (`sqrt`, `pi`, single-letter variables), no `__`, no attribute access, length ≤ 120 and expression size ≤ 60 ops. A test confirms that `__import__('os')…` never reaches the parser.

## Code correctness

Tests are hidden from the prompt. There are two formats:

* **I/O pairs** `{"input": [...], "output": ...}` with `entry_point`. These are preferred for RL. Comparison is **type-strict**: builtin types only, int/float compared with tolerance, containers compared recursively.
* **Assert strings** such as `"assert f(1) == 2"`, for MBPP/HumanEval compatibility.

The reward is 1 only if all tests pass. `code_partial_credit=True` gives the pass fraction, which rewards solutions that special-case the easy tests.

### Sandbox (defense in depth, not a security boundary)

| layer | blocks |
|---|---|
| static AST screen | non-whitelisted imports; `eval`/`exec`/`open`/`getattr`/`__import__`/…; **any** dunder attribute (`().__class__.__subclasses__()`); frame/code attributes (`gi_frame`, `f_back`, `f_globals`, …); string constants containing `__` (`"{0.__class__}".format`) |
| separate interpreter | `python -I -S` in a temp dir, minimal env, wall-clock timeout; POSIX `RLIMIT_AS`/`RLIMIT_CPU` |
| restricted builtins | whitelisted `__builtins__`, guarded `__import__`, no `SystemExit`/`exit` |
| unforgeable verdict | user stdout/stderr captured; the verdict goes on a line prefixed with a random per-call nonce |
| type-strict comparison | defeats `__eq__`-always-True objects |

For RL at scale on untrusted code, wrap `run_tests` in a container or gVisor sandbox with no network. The interface stays the same.

## Hacking detectors (`rewards/hacking.py`)

| flag | detector | guarded? | fix (see PAPER.md §6) |
|---|---|---|---|
| `empty_answer` | answer tag present but empty | yes | strict format needs non-empty content |
| `multiple_answers` | more than one distinct boxed or tagged candidate | yes | take the last tag only, and zero the reward |
| `prompt_copy` | ≥ 60% of the question's 5-grams reproduced | yes | strict extraction plus guard |
| `constant_answer` | one answer shared by ≥ 50% of distinct prompts in the batch (and more than 2× its gold frequency) | yes | batch-level guard |
| `eq_override` | class defines `__eq__`/`__ne__` | yes | type-strict comparison |
| `hardcoded_outputs` | ≥ 3 `if x == lit: return lit` branches, or a literal lookup table | yes | hidden tests plus guard |
| `sandbox_escape` | static-screen violation | yes | the code is never executed |
| `early_exit` | `raise SystemExit` / `exit()` | yes | the runner catches `BaseException` |
| `degenerate_repetition` | distinct 4-gram ratio < 0.3 | monitor only | KL penalty / length budget |

Monitor-only flags are logged as `hack/<flag>` but leave the reward unchanged, because a false positive there would penalize honest long reasoning. Every guarded flag has a unit test that reproduces the hack and checks that it earns zero reward (`tests/test_hacking_detection.py`).
