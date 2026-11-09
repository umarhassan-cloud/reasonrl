"""End-to-end stage runners shared by the CLI scripts and the ablation harness."""

from __future__ import annotations

import json
from pathlib import Path

from reasonrl.config import ExperimentConfig, save_config
from reasonrl.data.dataset import SFTDataset, load_examples, write_jsonl
from reasonrl.eval.benchmarks import EvalResult, compare, evaluate_policy, markdown_table
from reasonrl.eval.contamination import contamination_for_examples, split_clean
from reasonrl.logging_utils import MetricsLogger, get_logger
from reasonrl.models.load import load_model_and_tokenizer, make_reference_model, resolve_device
from reasonrl.models.lora import apply_lora, count_parameters
from reasonrl.rewards.composite import build_reward_fn
from reasonrl.rl.dpo import DPOTrainer, generate_preference_pairs
from reasonrl.rl.grpo import GRPOTrainer
from reasonrl.seed import set_seed
from reasonrl.sft.trainer import SFTTrainer

log = get_logger("reasonrl.pipeline")


def _run_dir(cfg: ExperimentConfig, stage: str) -> Path:
    return Path(cfg.output_dir) / cfg.name.replace("/", "__") / stage


def _load_eval_examples(cfg: ExperimentConfig, clean: bool = True):
    train = load_examples(cfg.data.train_files)
    ev = load_examples(cfg.data.eval_files, cfg.data.max_eval_examples)
    if not clean:
        return ev, None
    c = cfg.eval.contamination
    report = contamination_for_examples(train, ev, c.ngram, c.threshold, c.mask_numbers)
    clean_ev, dirty = split_clean(ev, report)
    if dirty:
        log.info(f"contamination: excluding {len(dirty)}/{len(ev)} eval items: {[e.id for e in dirty]}")
    return clean_ev, report


def _eval_fn(cfg, model, tok, examples, reward_fn):
    def fn() -> dict:
        res = evaluate_policy(
            model, tok, examples, reward_fn, n_samples=1, temperature=0.0,
            max_new_tokens=cfg.eval.max_new_tokens, batch_size=cfg.eval.batch_size,
            max_prompt_len=cfg.data.max_prompt_len, use_chat_template=cfg.model.use_chat_template,
            bootstrap_samples=200,
        )
        model.eval()
        return {f"eval/{k}": v for k, v in res.metrics.items() if not k.endswith(("_lo", "_hi"))}

    return fn


def run_sft(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    model, tok = load_model_and_tokenizer(cfg.model, device, cfg.sft.init_from)
    model = apply_lora(model, cfg.lora)
    tr, tot = count_parameters(model)
    run_dir = _run_dir(cfg, "sft")
    logger = MetricsLogger(run_dir, cfg.to_dict(), cfg.log_every, "reasonrl.sft")
    logger.info(f"trainable params {tr:,} / {tot:,}")
    train = load_examples(cfg.data.train_files, cfg.data.max_train_examples)
    ev, _ = _load_eval_examples(cfg)
    ds = SFTDataset(train, tok, cfg.data.max_prompt_len, cfg.data.max_target_len, cfg.model.use_chat_template)
    eds = SFTDataset([e for e in ev if e.solution or e.answer], tok, cfg.data.max_prompt_len,
                     cfg.data.max_target_len, cfg.model.use_chat_template)
    trainer = SFTTrainer(model, tok, ds, cfg, logger, eds if len(eds) else None, device)
    trainer.train()
    out = trainer.save(cfg.sft.save_dir)
    save_config(cfg, out / "experiment.yaml")
    logger.close()
    return out


def run_grpo(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    policy, tok = load_model_and_tokenizer(cfg.model, device, cfg.grpo.init_from)
    ref = make_reference_model(policy) if cfg.grpo.kl_coef > 0 else None
    policy = apply_lora(policy, cfg.lora)
    run_dir = _run_dir(cfg, "grpo")
    logger = MetricsLogger(run_dir, cfg.to_dict(), cfg.log_every, "reasonrl.grpo")
    reward_fn = build_reward_fn(cfg.reward)
    train = load_examples(cfg.data.train_files, cfg.data.max_train_examples)
    ev, _ = _load_eval_examples(cfg)
    trainer = GRPOTrainer(
        policy, ref, tok, train, reward_fn, cfg, logger, ev, device,
        eval_fn=_eval_fn(cfg, policy, tok, ev, reward_fn) if ev else None,
    )
    trainer.train()
    out = trainer.save(cfg.grpo.save_dir)
    save_config(cfg, out / "experiment.yaml")
    write_jsonl(run_dir / "samples.jsonl", trainer.samples_log)
    logger.close()
    return out


def run_dpo(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    policy, tok = load_model_and_tokenizer(cfg.model, device, cfg.dpo.init_from)
    ref = make_reference_model(policy)
    run_dir = _run_dir(cfg, "dpo")
    logger = MetricsLogger(run_dir, cfg.to_dict(), cfg.log_every, "reasonrl.dpo")
    reward_fn = build_reward_fn(cfg.reward)
    train = load_examples(cfg.data.train_files, cfg.data.max_train_examples)
    pairs_path = Path(cfg.dpo.pairs_file) if cfg.dpo.pairs_file else run_dir / "pairs.jsonl"
    if pairs_path.exists():
        pairs = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logger.info(f"loaded {len(pairs)} cached pairs from {pairs_path}")
    else:
        pairs, stats = generate_preference_pairs(policy, tok, train, reward_fn, cfg)
        logger.log(0, {f"pairs/{k}": v for k, v in stats.items()})
        write_jsonl(pairs_path, pairs)
    if not pairs:
        raise RuntimeError(
            "no preference pairs: the SFT model is never (or always) correct on train prompts. "
            "Increase dpo.samples_per_prompt / temperature, or set dpo.gold_as_chosen=true."
        )
    policy = apply_lora(policy, cfg.lora)
    trainer = DPOTrainer(policy, ref, tok, pairs, cfg, logger, device)
    trainer.train()
    out = trainer.save(cfg.dpo.save_dir)
    save_config(cfg, out / "experiment.yaml")
    logger.close()
    return out


def run_eval(cfg: ExperimentConfig) -> dict:
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    reward_fn = build_reward_fn(cfg.reward)
    examples, report = _load_eval_examples(cfg, clean=cfg.eval.use_clean_subset)
    checkpoints = cfg.eval.checkpoints or {"model": cfg.model.name_or_path}
    results: dict[str, EvalResult] = {}
    for name, path in checkpoints.items():
        local_like = path.startswith((".", "/", "runs")) or "\\" in path or len(Path(path).parts) > 2
        if local_like and not Path(path).exists():
            log.warning(f"skipping checkpoint '{name}': {path} does not exist")
            continue
        set_seed(cfg.seed)  # identical sampling noise for every checkpoint
        model, tok = load_model_and_tokenizer(cfg.model, device, path)
        res = evaluate_policy(
            model, tok, examples, reward_fn, name=name, n_samples=cfg.eval.n_samples,
            k_values=cfg.eval.k_values, temperature=cfg.eval.temperature, top_p=cfg.eval.top_p,
            max_new_tokens=cfg.eval.max_new_tokens, batch_size=cfg.eval.batch_size,
            max_prompt_len=cfg.data.max_prompt_len, use_chat_template=cfg.model.use_chat_template,
            bootstrap_samples=cfg.eval.bootstrap_samples,
        )
        results[name] = res
        log.info(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in res.metrics.items()))
        del model
    baseline = cfg.eval.baseline or next(iter(results))
    rows = compare(results, baseline, cfg.eval.bootstrap_samples)
    table = markdown_table(rows)
    out = {
        "protocol": {
            "n_eval": len(examples),
            "clean_subset": cfg.eval.use_clean_subset,
            "temperature": cfg.eval.temperature,
            "n_samples": cfg.eval.n_samples,
            "max_new_tokens": cfg.eval.max_new_tokens,
            "seed": cfg.seed,
        },
        "contamination": report.to_dict() if report else None,
        "rows": rows,
        "per_example": {k: v.per_example for k, v in results.items()},
        "table_markdown": table,
    }
    path = Path(cfg.eval.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("\n" + table)
    return out
