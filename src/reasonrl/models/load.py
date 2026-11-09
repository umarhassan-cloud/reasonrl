"""Model / tokenizer loading.

``name_or_path == "tiny-random"`` builds a randomly initialized 2-layer GPT-2
with a character-level tokenizer: no network access, runs on CPU in seconds.
Every code path (SFT, GRPO, DPO, eval) is exercised on it in tests. Any other
value is passed to ``transformers.AutoModelForCausalLM.from_pretrained``.
"""

from __future__ import annotations

import copy
import json
import string
from pathlib import Path
from typing import Any

import torch

from reasonrl.config import ModelConfig

CHAR_TOKENIZER_FILE = "char_tokenizer.json"

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


class CharTokenizer:
    """Character-level tokenizer exposing the subset of the HF API we use."""

    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, chars: str | None = None) -> None:
        chars = chars if chars is not None else string.printable
        self.chars = "".join(dict.fromkeys(chars))
        self.itos = list(self.SPECIALS) + list(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2, 3
        self.pad_token, self.bos_token, self.eos_token = "<pad>", "<bos>", "<eos>"
        self.chat_template = None
        self.padding_side = "right"

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.stoi.get(c, self.unk_token_id) for c in text]
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        n_special = len(self.SPECIALS)
        out = []
        for i in ids:
            if i < n_special:
                if not skip_special_tokens:
                    out.append(self.itos[i])
                continue
            if i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)

    def batch_decode(self, batch: Any, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(x, skip_special_tokens) for x in batch]

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / CHAR_TOKENIZER_FILE, "w", encoding="utf-8") as fh:
            json.dump({"chars": self.chars}, fh)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> CharTokenizer:
        with open(Path(path) / CHAR_TOKENIZER_FILE, encoding="utf-8") as fh:
            return cls(json.load(fh)["chars"])


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def build_tiny_model(
    vocab_size: int,
    n_embd: int = 64,
    n_layer: int = 2,
    n_head: int = 2,
    n_positions: int = 1024,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
) -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    cfg._attn_implementation = "eager"  # deterministic masking with left padding on CPU
    return GPT2LMHeadModel(cfg)


def _torch_dtype(name: str, device: torch.device) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    dtype = _DTYPES[name]
    if device.type == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def load_model_and_tokenizer(
    cfg: ModelConfig, device: torch.device | str = "cpu", path_override: str | None = None
) -> tuple[torch.nn.Module, Any]:
    device = torch.device(device) if isinstance(device, str) else device
    name = path_override or cfg.name_or_path
    if name == "tiny-random":
        tok = CharTokenizer()
        model = build_tiny_model(tok.vocab_size, **cfg.tiny)
        return model.to(device), tok

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (Path(name) / CHAR_TOKENIZER_FILE).exists():
        tok = CharTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        model.config._attn_implementation = "eager"
        return model.to(device), tok

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    dtype = _torch_dtype(cfg.dtype, device)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model.to(device), tok


def make_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Frozen copy of the policy at initialization (the SFT model)."""
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, path: str | Path) -> Path:
    from reasonrl.models.lora import merge_lora

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    to_save = merge_lora(copy.deepcopy(model)) if _has_lora(model) else model
    to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def _has_lora(model: torch.nn.Module) -> bool:
    from reasonrl.models.lora import LoRALinear

    return any(isinstance(m, LoRALinear) for m in model.modules()) or hasattr(model, "peft_config")
