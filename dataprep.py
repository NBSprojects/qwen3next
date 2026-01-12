import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

from dataclasses import dataclass
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from configs import TrainConfig

class TokenBlockDataset(Dataset):
    """
    Dataset qui prend un gros vecteur 1D de token ids et le découpe
    en blocks de taille fixe block_size.
    """

    def __init__(self, all_ids: torch.Tensor, block_size: int):
        assert all_ids.dtype == torch.long
        self.block_size = block_size

        # Troncature pour être multiple de block_size
        n_blocks = all_ids.numel() // block_size
        all_ids = all_ids[: n_blocks * block_size]

        self.data = all_ids.view(n_blocks, block_size)  # [n_blocks, block_size]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        # retour : [block_size]
        return self.data[idx]


def build_token_datasets(cfg: TrainConfig, tokenizer: GPT2TokenizerFast):
    """
    Charge WikiText-103, tokenize avec GPT-2 BPE, et renvoie
    deux TokenBlockDataset (train/val) avec block_size fixe.
    """
    print("[INFO] Chargement du dataset WikiText-103...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)

    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]

    def tokenize_split(texts: List[str]) -> torch.Tensor:
        all_ids: List[int] = []
        for t in texts:
            t = t.strip()
            if not t:
                continue
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 0:
                continue
            all_ids.extend(ids)
        return torch.tensor(all_ids, dtype=torch.long)

    print("[INFO] Tokenization train...")
    train_ids = tokenize_split(train_texts)
    print(f"[INFO] Nombre total de tokens train : {train_ids.numel():,}")

    print("[INFO] Tokenization val...")
    val_ids = tokenize_split(val_texts)
    print(f"[INFO] Nombre total de tokens val   : {val_ids.numel():,}")

    train_dataset = TokenBlockDataset(train_ids, cfg.block_size)
    val_dataset = TokenBlockDataset(val_ids, cfg.block_size)

    print(f"[INFO] Nombre de séquences train : {len(train_dataset)}")
    print(f"[INFO] Nombre de séquences val   : {len(val_dataset)}")

    return train_dataset, val_dataset