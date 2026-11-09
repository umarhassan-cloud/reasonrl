import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasonrl.models.load import CharTokenizer, build_tiny_model  # noqa: E402


@pytest.fixture
def tokenizer():
    return CharTokenizer()


@pytest.fixture
def tiny_model(tokenizer):
    torch.manual_seed(0)
    return build_tiny_model(tokenizer.vocab_size, n_embd=32, n_layer=1, n_head=2, n_positions=512)


@pytest.fixture
def repo_root():
    return ROOT
