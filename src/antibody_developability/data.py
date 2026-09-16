from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB = {PAD_TOKEN: 0, UNK_TOKEN: 1}
VOCAB.update({aa: idx + 2 for idx, aa in enumerate(AMINO_ACIDS)})


def encode_sequence(sequence: str, max_len: int) -> torch.Tensor:
    """Encode an amino-acid sequence as integer token IDs."""
    sequence = str(sequence).strip().upper()
    tokens = [VOCAB.get(aa, VOCAB[UNK_TOKEN]) for aa in sequence[:max_len]]
    tokens += [VOCAB[PAD_TOKEN]] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


@dataclass(frozen=True)
class AntibodyExample:
    heavy_chain: str
    light_chain: str
    label: float


class AntibodyDataset(Dataset):
    """PyTorch dataset for paired antibody chains."""

    REQUIRED_COLUMNS = {"heavy_chain", "light_chain", "label"}

    def __init__(self, csv_path: str | Path, max_len: int = 256):
        self.csv_path = Path(csv_path)
        self.max_len = max_len
        self.df = pd.read_csv(self.csv_path)

        missing = self.REQUIRED_COLUMNS.difference(self.df.columns)
        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required columns: {sorted(missing)}"
            )

        if self.df.empty:
            raise ValueError(f"{self.csv_path} contains no rows.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        heavy = encode_sequence(row["heavy_chain"], self.max_len)
        light = encode_sequence(row["light_chain"], self.max_len)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "heavy": heavy,
            "light": light,
            "label": label,
        }


def load_split_csvs(data_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        split: pd.read_csv(data_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
