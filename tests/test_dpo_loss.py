import math

import pytest
import torch
import torch.nn.functional as F

from reasonrl.config import ExperimentConfig
from reasonrl.data.collators import PreferenceCollator
from reasonrl.data.dataset import Example, PreferenceDataset
from reasonrl.logging_utils import MetricsLogger
from reasonrl.models.load import make_reference_model
from reasonrl.rewards.composite import RewardOutput
from reasonrl.rl.dpo import DPOTrainer, build_preference_pairs, dpo_loss
from reasonrl.rl.rollout import sequence_logps


def test_dpo_loss_at_init_is_log2():
    z = torch.zeros(4)
    loss, m = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert m["rewards/margin"] == 0.0


def test_dpo_loss_matches_closed_form():
    pc, pr = torch.tensor([-1.0, -3.0]), torch.tensor([-2.0, -1.0])
    rc, rr = torch.tensor([-1.5, -2.0]), torch.tensor([-1.5, -2.0])
    beta = 0.5
    loss, m = dpo_loss(pc, pr, rc, rr, beta=beta)
    logits = (pc - pr) - (rc - rr)
    assert loss.item() == pytest.approx((-F.logsigmoid(beta * logits)).mean().item())
    assert m["rewards/accuracy"] == pytest.approx(0.5)
    assert m["rewards/chosen"] == pytest.approx((beta * (pc - rc)).mean().item())


def test_dpo_gradient_direction():
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-2.0], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.1)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # descent raises chosen, lowers rejected


def test_dpo_variants():
    pc, pr, rc, rr = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
    ipo, _ = dpo_loss(pc, pr, rc, rr, beta=0.1, loss_type="ipo")
    assert ipo.item() == pytest.approx((1 / 0.2) ** 2)
    hinge, _ = dpo_loss(pc, pr, rc, rr, beta=0.1, loss_type="hinge")
    assert hinge.item() == pytest.approx(1.0)
    smooth, _ = dpo_loss(pc + 5, pr, rc, rr, beta=1.0, label_smoothing=0.1)
    plain, _ = dpo_loss(pc + 5, pr, rc, rr, beta=1.0)
    assert smooth.item() > plain.item()
    with pytest.raises(ValueError):
        dpo_loss(pc, pr, rc, rr, loss_type="nope")


def _ro(correct: float, guarded: bool = False, total: float | None = None) -> RewardOutput:
    return RewardOutput(total=correct if total is None else total, correctness=correct, format=1.0,
                        extracted=None, guarded=guarded)


def test_pair_construction_correct_vs_incorrect():
    ex = [Example(id="a", task="math", question="q", answer="1", solution="<answer>1</answer>"),
          Example(id="b", task="math", question="q2", answer="2")]
    comps = [["right1", "wrong1", "hacked"], ["w1", "w2"]]
    scores = [[_ro(1.0), _ro(0.0), _ro(1.0, guarded=True)], [_ro(0.0), _ro(0.0)]]
    pairs = build_preference_pairs(ex, comps, scores, ["P1", "P2"], pairs_per_prompt=2)
    assert {p["id"] for p in pairs} == {"a"}  # prompt b has no correct sample
    assert all(p["chosen"] == "right1" for p in pairs)
    assert pairs[0]["rejected"] == "hacked"  # hacked completions are rejected first, never chosen
    with_gold = build_preference_pairs(ex, comps, scores, ["P1", "P2"], pairs_per_prompt=1, gold_as_chosen=True)
    assert {p["id"] for p in with_gold} == {"a"}  # b has no solution field -> still skipped


def test_pair_construction_best_vs_worst():
    ex = [Example(id="a", task="math", question="q", answer="1")]
    pairs = build_preference_pairs(ex, [["x", "y", "z", "w"]], [[_ro(0, total=0.1), _ro(1, total=1.1),
                                    _ro(0, total=0.0), _ro(0, total=0.05)]], ["P"], strategy="best_vs_worst")
    assert pairs[0]["chosen"] == "y" and pairs[0]["rejected"] == "z"


def test_sequence_logps_masks_prompt(tiny_model, tokenizer):
    ds = PreferenceDataset([{"prompt": "Q: hi", "chosen": "good", "rejected": "bad!!"}], tokenizer)
    batch = PreferenceCollator(tokenizer.pad_token_id)([ds[0]])
    lp = sequence_logps(tiny_model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    assert lp.shape == (2,) and (lp < 0).all()
    # chosen = "good"+eos (5 tokens), rejected = "bad!!"+eos (6 tokens): label counts differ
    assert int((batch["labels"][0] != -100).sum()) == 5 and int((batch["labels"][1] != -100).sum()) == 6


def test_dpo_trainer_increases_margin(tiny_model, tokenizer, tmp_path):
    cfg = ExperimentConfig()
    cfg.optim.lr = 5e-3
    cfg.optim.scheduler = "constant"
    cfg.dpo.epochs, cfg.dpo.batch_size, cfg.dpo.beta = 15, 2, 0.5
    pairs = [{"id": str(i), "prompt": f"Q{i}: 2+2=", "chosen": "<answer>4</answer>", "rejected": "<answer>5</answer>"}
             for i in range(4)]
    ref = make_reference_model(tiny_model)
    tr = DPOTrainer(tiny_model, ref, tokenizer, pairs, cfg, MetricsLogger(tmp_path))
    hist = tr.train()
    assert hist[0]["train/loss"] == pytest.approx(math.log(2), abs=1e-3)  # policy == ref at step 1
    assert hist[-1]["train/rewards/margin"] > hist[0]["train/rewards/margin"]
    assert hist[-1]["train/loss"] < hist[0]["train/loss"]
