"""Sampling with KV cache and log-probability computation.

We implement our own sampling loop instead of ``model.generate`` so that the
exact token ids, masks and position ids used at sampling time are the ones the
policy-gradient loss is computed on (no re-tokenization, no hidden logits
processors), and so the code works identically for HF and custom tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reasonrl.data.collators import left_pad


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp_(min=0)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(-1)
    cum = probs.cumsum(-1)
    remove = cum - probs > top_p  # keep the token that crosses the threshold
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    logits = top_p_filter(logits.float() / temperature, top_p)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


@dataclass
class Generation:
    input_ids: torch.Tensor  # [B, P+L] prompt (left padded) + completion (right padded)
    attention_mask: torch.Tensor  # [B, P+L]
    action_mask: torch.Tensor  # [B, P+L-1] float, 1 on completion tokens
    completion_ids: list  # list[list[int]] without padding (EOS kept if produced)
    prompt_len: int
    truncated: torch.Tensor  # [B] bool, hit max_new_tokens without EOS


@torch.no_grad()
def sample_completions(
    model: torch.nn.Module,
    prompt_ids: list[list[int]],
    pad_id: int,
    eos_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Generation:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids, attn = left_pad(prompt_ids, pad_id)
    ids, attn = ids.to(device), attn.to(device)
    B, P = ids.shape
    pos = position_ids_from_mask(attn)
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    next_pos = pos[:, -1:] + 1
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    gen_tokens, gen_real = [], []
    cur_attn = attn
    for _ in range(max_new_tokens):
        tok = sample_next(logits, temperature, top_p)
        real = ~finished
        tok = torch.where(real, tok, torch.full_like(tok, pad_id))
        gen_tokens.append(tok)
        gen_real.append(real)
        finished = finished | (tok == eos_id)
        if bool(finished.all()):
            break
        cur_attn = torch.cat([cur_attn, real.long()[:, None]], dim=1)
        out = model(
            input_ids=tok[:, None],
            attention_mask=cur_attn,
            position_ids=next_pos,
            past_key_values=past,
            use_cache=True,
        )
        past, logits = out.past_key_values, out.logits[:, -1]
        next_pos = next_pos + 1
    if was_training:
        model.train()
    gen = torch.stack(gen_tokens, 1) if gen_tokens else torch.empty(B, 0, dtype=torch.long, device=device)
    gmask = torch.stack(gen_real, 1).long() if gen_real else torch.empty(B, 0, dtype=torch.long, device=device)
    full_ids = torch.cat([ids, gen], dim=1)
    full_attn = torch.cat([attn, gmask], dim=1)
    action_mask = torch.zeros(B, full_ids.size(1) - 1, device=device)
    if gen.size(1):
        action_mask[:, P - 1 :] = gmask.float()
    completions = [gen[i][gmask[i].bool()].tolist() for i in range(B)]
    return Generation(full_ids, full_attn, action_mask, completions, P, ~finished)


def token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """``log pi(x_{t+1} | x_{<=t})`` for every position -> ``[B, T-1]`` (fp32).

    Uses ``gather - logsumexp`` to avoid materializing a full log-softmax copy.
    """
    pos = position_ids_from_mask(attention_mask)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos).logits
    logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def sequence_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Sum of completion-token log-probs per sequence (labels != -100) -> ``[B]``."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = (tgt != -100).float()
    safe = tgt.clamp(min=0)
    lp = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return (lp * mask).sum(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)
