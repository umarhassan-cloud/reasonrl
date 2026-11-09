"""Padding / batching."""

from __future__ import annotations

import torch


def pad_sequences(seqs: list[list[int]], pad_value: int, side: str = "right") -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if not s:
            continue
        t = torch.tensor(s, dtype=torch.long)
        if side == "right":
            out[i, : len(s)] = t
        elif side == "left":
            out[i, max_len - len(s) :] = t
        else:
            raise ValueError(side)
    return out


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad prompts for batched generation -> (input_ids, attention_mask)."""
    ids = pad_sequences(seqs, pad_id, side="left")
    mask = pad_sequences([[1] * len(s) for s in seqs], 0, side="left")
    return ids, mask


class SFTCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        ids = [b["input_ids"] for b in batch]
        return {
            "input_ids": pad_sequences(ids, self.pad_id),
            "attention_mask": pad_sequences([[1] * len(s) for s in ids], 0),
            "labels": pad_sequences([b["labels"] for b in batch], -100),
        }


class PreferenceCollator:
    """Stacks chosen then rejected into one ``[2B, T]`` batch (single forward pass)."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        seqs, labels = [], []
        for key in ("chosen_ids", "rejected_ids"):
            for b in batch:
                p, r = b["prompt_ids"], b[key]
                seqs.append(p + r)
                labels.append([-100] * len(p) + r)
        return {
            "input_ids": pad_sequences(seqs, self.pad_id),
            "attention_mask": pad_sequences([[1] * len(s) for s in seqs], 0),
            "labels": pad_sequences(labels, -100),
            "n_pairs": len(batch),
        }
