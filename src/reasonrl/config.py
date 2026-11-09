"""Typed experiment configuration.

Configs are YAML files mapped onto nested dataclasses. Unknown keys raise
errors (typos in an ablation config should fail loudly, not silently run the
baseline). A config may inherit from another via ``base: <relative path>`` and
can be patched from the CLI with dotted overrides such as
``grpo.kl_coef=0.0``.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    use_chat_template: bool = False
    attn_implementation: str | None = None
    # Only used when name_or_path == "tiny-random" (CPU tests / smoke runs).
    tiny: dict = field(
        default_factory=lambda: {"n_embd": 64, "n_layer": 2, "n_head": 2, "n_positions": 1024}
    )


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
    )
    backend: str = "native"  # native | peft


@dataclass
class DataConfig:
    train_files: list = field(default_factory=lambda: ["data/sample/math_train.jsonl"])
    eval_files: list = field(default_factory=lambda: ["data/sample/math_eval.jsonl"])
    max_prompt_len: int = 384
    max_target_len: int = 512
    max_train_examples: int | None = None
    max_eval_examples: int | None = None


@dataclass
class OptimConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"  # cosine | linear | constant


@dataclass
class RewardConfig:
    correctness_weight: float = 1.0
    format_weight: float = 0.1
    format_mode: str = "strict"  # strict | naive  (naive is the hackable variant)
    require_format_for_correctness: bool = False
    strict_extraction: bool = True  # False -> fall back to "last number" (hackable)
    hacking_guards: bool = True
    hacking_penalty: float = 0.0  # guarded samples receive -penalty
    math_tolerance: float = 1e-6
    code_timeout_s: float = 3.0
    code_partial_credit: bool = False
    code_workers: int = 4
    constant_answer_threshold: float = 0.5
    prompt_copy_threshold: float = 0.6


@dataclass
class SFTConfig:
    epochs: int = 2
    batch_size: int = 8
    grad_accum: int = 1
    max_steps: int | None = None
    eval_every: int = 50
    save_dir: str = "runs/sft"
    init_from: str | None = None


@dataclass
class GRPOConfig:
    steps: int = 200
    prompts_per_step: int = 8
    group_size: int = 8
    max_new_tokens: int = 384
    temperature: float = 1.0
    top_p: float = 1.0
    ppo_epochs: int = 1
    minibatch_size: int = 16
    clip_eps: float = 0.2
    clip_eps_high: float | None = None  # DAPO "clip-higher"; None -> symmetric
    kl_coef: float = 0.04
    kl_estimator: str = "k3"  # k1 | k2 | k3
    adv_mode: str = "std"  # std (GRPO) | mean_only (Dr. GRPO) | rloo
    loss_agg: str = "seq_mean"  # seq_mean (DeepSeekMath) | token_mean (DAPO) | constant (Dr. GRPO)
    filter_zero_std_groups: bool = False
    eval_every: int = 50
    save_every: int | None = None
    save_dir: str = "runs/grpo"
    init_from: str | None = None


@dataclass
class DPOConfig:
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid | ipo | hinge
    label_smoothing: float = 0.0
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 1
    samples_per_prompt: int = 8  # matched to GRPO group_size for a fair sample budget
    pairs_per_prompt: int = 2
    pair_strategy: str = "correct_vs_incorrect"  # correct_vs_incorrect | best_vs_worst
    gold_as_chosen: bool = False
    max_new_tokens: int = 384
    temperature: float = 1.0
    top_p: float = 1.0
    pairs_file: str | None = None  # reuse cached pairs if present
    save_dir: str = "runs/dpo"
    init_from: str | None = None


@dataclass
class ContaminationConfig:
    ngram: int = 8
    threshold: float = 0.5
    mask_numbers: bool = False


@dataclass
class EvalConfig:
    checkpoints: dict = field(default_factory=dict)  # name -> path
    baseline: str | None = None
    n_samples: int = 1
    k_values: list = field(default_factory=lambda: [1])
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 384
    batch_size: int = 16
    use_clean_subset: bool = True
    bootstrap_samples: int = 2000
    output_path: str = "runs/eval/results.json"
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"
    output_dir: str = "runs"
    log_every: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# dict <-> dataclass
# --------------------------------------------------------------------------


def from_dict(cls: type, data: dict | None, path: str = "") -> Any:
    """Recursively build dataclass ``cls`` from ``data``; reject unknown keys."""
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config key(s) at '{path or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints[f.name]
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = from_dict(ftype, value, f"{path}.{f.name}".lstrip("."))
        kwargs[f.name] = value
    return cls(**kwargs)


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict:
    """``["grpo.kl_coef=0", "name=x"]`` -> nested dict (values parsed as YAML)."""
    patch: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_dotted(patch, key.strip(), yaml.safe_load(raw))
    return patch


def dotted_to_nested(flat: dict) -> dict:
    nested: dict = {}
    for key, value in flat.items():
        _set_dotted(nested, key, value)
    return nested


def load_raw_yaml(path: str | Path) -> dict:
    """Load YAML and resolve ``base:`` inheritance (relative to the file)."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)
    if base is not None:
        raw = deep_merge(load_raw_yaml(path.parent / base), raw)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = load_raw_yaml(path) if path else {}
    raw.pop("variants", None)
    raw.pop("description", None)
    raw = deep_merge(raw, parse_overrides(overrides))
    return from_dict(ExperimentConfig, raw)


def load_ablation(path: str | Path, overrides: list[str] | None = None) -> list[tuple[str, ExperimentConfig]]:
    """An ablation file = a base config plus ``variants: {name: {dotted.key: value}}``."""
    raw = load_raw_yaml(path)
    variants = raw.pop("variants", None) or {"default": {}}
    raw.pop("description", None)
    out = []
    for name, patch in variants.items():
        merged = deep_merge(raw, dotted_to_nested(patch or {}))
        merged = deep_merge(merged, parse_overrides(overrides))
        merged["name"] = f"{merged.get('name', 'ablation')}/{name}"
        out.append((name, from_dict(ExperimentConfig, merged)))
    return out


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
