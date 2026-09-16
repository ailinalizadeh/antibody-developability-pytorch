import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from antibody_developability.data import VOCAB, encode_sequence


def test_encode_sequence_has_fixed_length():
    encoded = encode_sequence("ACDE", max_len=10)
    assert encoded.shape == (10,)
    assert encoded.dtype == torch.long


def test_encode_sequence_uses_padding():
    encoded = encode_sequence("AC", max_len=5)
    assert encoded[-1].item() == VOCAB["<PAD>"]


def test_unknown_character_maps_to_unknown_token():
    encoded = encode_sequence("AZ", max_len=2)
    assert encoded[1].item() == VOCAB["<UNK>"]
