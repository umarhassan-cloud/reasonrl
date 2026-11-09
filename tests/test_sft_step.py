import torch

from reasonrl.config import ExperimentConfig, LoRAConfig, OptimConfig, load_ablation, load_config
from reasonrl.data.collators import SFTCollator, left_pad
from reasonrl.data.dataset import Example, SFTDataset, build_prompt_text
from reasonrl.logging_utils import MetricsLogger
from reasonrl.models.load import CharTokenizer, load_model_and_tokenizer, save_checkpoint
from reasonrl.models.lora import LoRALinear, apply_lora, count_parameters, merge_lora
from reasonrl.seed import set_seed
from reasonrl.sft.trainer import SFTTrainer, build_optimizer, build_scheduler, causal_lm_loss, sft_step

EXAMPLES = [
    Example(id=f"m{i}", task="math", question=f"What is {i} + {i}?", answer=str(2 * i),
            solution=f"<think>{i} plus {i} is {2 * i}</think>\n<answer>{2 * i}</answer>")
    for i in range(6)
]


def test_labels_mask_prompt_tokens(tokenizer):
    ds = SFTDataset(EXAMPLES[:1], tokenizer, max_prompt_len=512)
    item = ds[0]
    n_prompt = len(tokenizer.encode(build_prompt_text(EXAMPLES[0])))
    assert all(lab == -100 for lab in item["labels"][:n_prompt])
    assert item["labels"][n_prompt:] == item["input_ids"][n_prompt:]
    assert item["input_ids"][-1] == tokenizer.eos_token_id


def test_collators_pad_correctly():
    batch = SFTCollator(0)([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
    ids, mask = left_pad([[1, 2], [3]], 0)
    assert ids.tolist() == [[1, 2], [0, 3]] and mask.tolist() == [[1, 1], [0, 1]]


def test_sft_step_reduces_loss(tiny_model, tokenizer):
    set_seed(0)
    ds = SFTDataset(EXAMPLES, tokenizer, max_prompt_len=512)
    batch = SFTCollator(tokenizer.pad_token_id)([ds[i] for i in range(len(ds))])
    opt = build_optimizer(tiny_model, OptimConfig(lr=3e-3))
    first = sft_step(tiny_model, batch, opt)["loss"]
    for _ in range(25):
        last = sft_step(tiny_model, batch, opt)["loss"]
    assert last < first * 0.8


def test_scheduler_warmup_and_cosine():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_scheduler(opt, OptimConfig(lr=1.0, warmup_steps=2, scheduler="cosine"), total_steps=10)
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == 0.5 and lrs[1] == 1.0 and lrs[-1] < 0.1


def test_sft_trainer_and_checkpoint_roundtrip(tiny_model, tokenizer, tmp_path):
    cfg = ExperimentConfig()
    cfg.sft.max_steps, cfg.sft.batch_size, cfg.sft.eval_every = 4, 3, 2
    ds = SFTDataset(EXAMPLES, tokenizer, max_prompt_len=512)
    tr = SFTTrainer(tiny_model, tokenizer, ds, cfg, MetricsLogger(tmp_path / "log"), eval_ds=ds)
    hist = tr.train()
    assert [h["step"] for h in hist if "train/loss" in h] == [1, 2, 3, 4]
    assert (tmp_path / "log" / "metrics.jsonl").exists() and (tmp_path / "log" / "env.json").exists()
    path = save_checkpoint(tiny_model, tokenizer, tmp_path / "ckpt")
    cfg.model.name_or_path = str(path)
    model2, tok2 = load_model_and_tokenizer(cfg.model, "cpu")
    assert isinstance(tok2, CharTokenizer)
    batch = SFTCollator(0)([ds[0]])
    tiny_model.eval()
    model2.eval()
    assert torch.allclose(causal_lm_loss(tiny_model, batch)[0], causal_lm_loss(model2, batch)[0], atol=1e-5)


def test_native_lora_trains_adapters_only_and_merges(tiny_model, tokenizer):
    model = apply_lora(tiny_model, LoRAConfig(enabled=True, r=4, alpha=8, target_modules=["c_attn"]))
    trainable, total = count_parameters(model)
    assert 0 < trainable < total
    assert any(isinstance(m, LoRALinear) for m in model.modules())
    ds = SFTDataset(EXAMPLES, tokenizer, max_prompt_len=512)
    batch = SFTCollator(0)([ds[i] for i in range(3)])
    opt = build_optimizer(model, OptimConfig(lr=1e-2))
    for _ in range(3):
        sft_step(model, batch, opt)
    model.eval()
    before = causal_lm_loss(model, batch)[0]
    merged = merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(before, causal_lm_loss(merged, batch)[0], atol=1e-4)


def test_config_loading_overrides_and_ablations(repo_root):
    cfg = load_config(repo_root / "configs" / "smoke" / "grpo_tiny.yaml", ["grpo.kl_coef=0.0", "seed=7"])
    assert cfg.model.name_or_path == "tiny-random" and cfg.grpo.kl_coef == 0.0 and cfg.seed == 7
    assert cfg.grpo.clip_eps == 0.2  # inherited from configs/grpo.yaml
    variants = dict(load_ablation(repo_root / "experiments" / "ablations" / "group_size.yaml"))
    assert variants["g4_budget64"].grpo.group_size * variants["g4_budget64"].grpo.prompts_per_step == 64
    assert variants["g8_budget64"].grpo.group_size == 8
    try:
        load_config(None, ["grpo.not_a_key=1"])
    except KeyError as e:
        assert "not_a_key" in str(e)
    else:
        raise AssertionError("unknown keys must raise")
