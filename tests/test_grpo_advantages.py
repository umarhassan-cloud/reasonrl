import pytest
import torch

from reasonrl.config import ExperimentConfig
from reasonrl.data.dataset import Example
from reasonrl.logging_utils import MetricsLogger
from reasonrl.models.load import make_reference_model
from reasonrl.rewards import RewardFunction
from reasonrl.rl.advantages import group_relative_advantages, zero_std_groups
from reasonrl.rl.buffer import RolloutBatch, RolloutBuffer
from reasonrl.rl.grpo import GRPOTrainer, aggregate, grpo_loss, kl_penalty
from reasonrl.rl.rollout import sample_completions, token_logprobs, top_p_filter


def test_std_advantages_are_group_normalized():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 6.0])
    adv = group_relative_advantages(r, group_size=4).view(2, 4)
    assert torch.allclose(adv.mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(adv.std(1), torch.ones(2), atol=1e-3)
    # group-relative: shifting one group's rewards doesn't change its advantages
    assert torch.allclose(group_relative_advantages(r + 100, 4), group_relative_advantages(r, 4), atol=1e-5)


def test_identical_rewards_give_zero_advantage():
    r = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    adv = group_relative_advantages(r, 3)
    assert torch.allclose(adv[:3], torch.zeros(3))
    assert zero_std_groups(r, 3).tolist() == [True] * 3 + [False] * 3


def test_rloo_equals_leave_one_out_baseline():
    r = torch.tensor([3.0, 1.0, 2.0, 6.0])
    adv = group_relative_advantages(r, 4, mode="rloo")
    expected = torch.stack([r[i] - torch.cat([r[:i], r[i + 1 :]]).mean() for i in range(4)])
    assert torch.allclose(adv, expected)
    assert torch.allclose(group_relative_advantages(r, 4, mode="mean_only"), r - r.mean())


def test_advantage_input_validation():
    with pytest.raises(ValueError):
        group_relative_advantages(torch.ones(5), 2)
    with pytest.raises(ValueError):
        group_relative_advantages(torch.ones(4), 1)


def test_kl_estimators():
    logp = torch.log(torch.tensor([0.5, 0.2]))
    ref = torch.log(torch.tensor([0.5, 0.4]))
    k3 = kl_penalty(logp, ref, "k3")
    assert k3[0].item() == pytest.approx(0.0, abs=1e-7) and (k3 >= 0).all()
    assert kl_penalty(logp, logp, "k1").abs().max() == 0
    with pytest.raises(ValueError):
        kl_penalty(logp, ref, "k9")


def test_grpo_loss_on_policy_gradient_matches_reinforce():
    # with ratio == 1 and no KL, grad of loss == -A * grad(logp) / |o| (seq_mean)
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.5]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([1.0, -2.0])
    loss, stats = grpo_loss(logp, logp.detach(), None, adv, mask, kl_coef=0.0, loss_agg="seq_mean")
    loss.backward()
    expected = torch.tensor([[-0.25, -0.25], [1.0, 0.0]])  # -A_i / (|o_i| * B)
    assert torch.allclose(logp.grad, expected)
    assert stats["clip_frac"] == 0.0 and stats["ratio_mean"] == pytest.approx(1.0)


def test_grpo_clipping_zeroes_gradient_outside_trust_region():
    old = torch.zeros(1, 1)
    logp = torch.full((1, 1), 0.5, requires_grad=True)  # ratio = e^0.5 > 1.2
    loss, stats = grpo_loss(logp, old, None, torch.tensor([1.0]), torch.ones(1, 1), clip_eps=0.2, kl_coef=0.0)
    loss.backward()
    assert logp.grad.abs().item() == 0.0 and stats["clip_frac"] == 1.0
    # clip-higher widens the upper bound
    logp2 = torch.full((1, 1), 0.1, requires_grad=True)  # ratio ~1.105
    loss2, _ = grpo_loss(logp2, old, None, torch.tensor([1.0]), torch.ones(1, 1), clip_eps=0.05, clip_eps_high=0.28,
                         kl_coef=0.0)
    loss2.backward()
    assert logp2.grad.abs().item() > 0


def test_kl_term_pulls_toward_reference():
    ref = torch.zeros(1, 3)
    logp = torch.full((1, 3), -0.3, requires_grad=True)
    loss, stats = grpo_loss(logp, logp.detach(), ref, torch.zeros(1), torch.ones(1, 3), kl_coef=1.0)
    loss.backward()
    assert stats["kl"] > 0
    assert (logp.grad < 0).all()  # descent increases logp toward ref


def test_loss_aggregation_modes():
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    assert aggregate(x, m, "seq_mean").item() == pytest.approx((1.0 + 4.0) / 2)
    assert aggregate(x, m, "token_mean").item() == pytest.approx(8.0 / 5)
    assert aggregate(x, m, "constant").item() == pytest.approx(8.0 / 8)


def test_top_p_filter_keeps_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept = torch.isfinite(top_p_filter(logits, 0.7))
    assert kept.tolist() == [[True, True, False, False]]


def test_sampling_masks_and_logprob_alignment(tiny_model, tokenizer):
    prompts = [tokenizer.encode("abc"), tokenizer.encode("hello world")]
    torch.manual_seed(0)
    gen = sample_completions(tiny_model, prompts, tokenizer.pad_token_id, tokenizer.eos_token_id, 12, 1.0)
    B, T = gen.input_ids.shape
    assert gen.action_mask.shape == (B, T - 1)
    assert gen.action_mask[:, : gen.prompt_len - 1].sum() == 0
    for i in range(B):
        assert int(gen.action_mask[i].sum()) == len(gen.completion_ids[i])
    # left padding (row 0 has the shorter prompt) must not change real-token logprobs
    lp = token_logprobs(tiny_model, gen.input_ids, gen.attention_mask)
    assert torch.isfinite(lp).all()
    mask = gen.attention_mask[0].bool()
    solo_ids = gen.input_ids[0][mask][None]
    lp_solo = token_logprobs(tiny_model, solo_ids, torch.ones_like(solo_ids))
    valid = mask[:-1] & mask[1:]
    assert torch.allclose(lp[0][valid], lp_solo[0], atol=1e-4)


def test_greedy_kv_cache_matches_full_recompute(tiny_model, tokenizer):
    prompt = tokenizer.encode("Problem: 1+1")
    gen = sample_completions(tiny_model, [prompt], tokenizer.pad_token_id, -1, 8, temperature=0.0)
    ids = torch.tensor([prompt])
    for _ in range(8):
        nxt = tiny_model(input_ids=ids).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    assert gen.input_ids.tolist() == ids.tolist()


def test_buffer_minibatches_cover_batch():
    b = RolloutBatch(*(torch.arange(10).view(10, 1).float() for _ in range(5)), advantages=torch.arange(10.0),
                     rewards=torch.zeros(10))
    buf = RolloutBuffer(b)
    seen = torch.cat([mb.advantages for mb in buf.iter_minibatches(3, generator=torch.Generator().manual_seed(0))])
    assert sorted(seen.tolist()) == list(range(10))


def _toy_examples():
    return [Example(id=f"m{i}", task="math", question=f"What is {i} + 1?", answer=str(i + 1)) for i in range(4)]


def test_grpo_trainer_step_updates_policy(tiny_model, tokenizer, tmp_path):
    cfg = ExperimentConfig()
    cfg.data.max_prompt_len = 256
    cfg.optim.lr = 1e-2
    g = cfg.grpo
    g.prompts_per_step, g.group_size, g.max_new_tokens, g.minibatch_size = 2, 4, 8, 8
    ref = make_reference_model(tiny_model)
    before = [p.detach().clone() for p in tiny_model.parameters()]

    # a reward that varies within groups so advantages are non-zero
    class LengthReward(RewardFunction):
        def score_batch(self, examples, completions):
            outs = super().score_batch(examples, completions)
            for o, c in zip(outs, completions):
                o.total = float(len(c) % 3)
            return outs

    trainer = GRPOTrainer(tiny_model, ref, tokenizer, _toy_examples(), LengthReward(), cfg,
                          MetricsLogger(tmp_path), device="cpu")
    torch.manual_seed(0)
    m = trainer.step()
    for key in ("reward/mean", "train/kl", "train/clip_frac", "completion/len_mean", "hack/empty_answer"):
        assert key in m
    changed = any(not torch.equal(a, b) for a, b in zip(before, tiny_model.parameters()))
    assert changed
    # reference is frozen
    assert all(not p.requires_grad for p in ref.parameters())


def test_grpo_requires_ref_when_kl_enabled(tiny_model, tokenizer, tmp_path):
    cfg = ExperimentConfig()
    with pytest.raises(ValueError):
        GRPOTrainer(tiny_model, None, tokenizer, _toy_examples(), RewardFunction(), cfg, MetricsLogger(tmp_path))
