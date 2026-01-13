import os
import time
import pyarrow as pa
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

from itertools import chain
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


def build_token_datasets_old(cfg: TrainConfig, tokenizer: GPT2TokenizerFast):
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
    start_train = time.time()
    train_ids = tokenize_split(train_texts)
    train_time = time.time() - start_train
    print(f"[INFO] Nombre total de tokens train : {train_ids.numel():,}")
    print(f"[INFO] Temps de tokenization train  : {train_time:.2f}s ({train_ids.numel() / train_time:.0f} tokens/s)")

    print("[INFO] Tokenization val...")
    start_val = time.time()
    val_ids = tokenize_split(val_texts)
    val_time = time.time() - start_val
    print(f"[INFO] Nombre total de tokens val   : {val_ids.numel():,}")
    print(f"[INFO] Temps de tokenization val    : {val_time:.2f}s ({val_ids.numel() / val_time:.0f} tokens/s)")

    train_dataset = TokenBlockDataset(train_ids, cfg.block_size)
    val_dataset = TokenBlockDataset(val_ids, cfg.block_size)

    print(f"[INFO] Nombre de séquences train : {len(train_dataset)}")
    print(f"[INFO] Nombre de séquences val   : {len(val_dataset)}")

    return train_dataset, val_dataset


def build_token_datasets(cfg: TrainConfig, tokenizer: GPT2TokenizerFast):
    """
    Version optimisée : utilise multiprocessing et batching pour accélérer
    la tokenization de WikiText-103.
    """
    print("[INFO] Chargement du dataset WikiText-103...")
    # On garde seulement les colonnes nécessaires pour éviter de saturer la RAM
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)

    def batch_tokenize(examples):
        # Le tokenizer gère ici une LISTE de textes d'un coup (beaucoup plus rapide en Rust)
        return tokenizer(examples["text"], add_special_tokens=False)

    print(f"[INFO] Tokenization en cours (CPUs: {os.cpu_count()})...")

    start_tokenization = time.time()
    
    # Étape 1 : Tokenization parallèle via .map()
    tokenized_ds = ds.map(
        batch_tokenize,
        batched=True,             # Active le traitement par lots
        num_proc=os.cpu_count(),  # Utilise tous les coeurs CPU disponibles
        remove_columns=["text"],  # Supprime le texte brut pour libérer la RAM immédiatement
        desc="Tokenization"
    )

    def flatten_to_tensor(dataset_split):
            """
            Version Ultra-Rapide : utilise le backend C++ (Arrow) pour aplatir
            les listes sans passer par des boucles Python.
            """
            print(f"[INFO] Fusion optimisée (Zero-Copy) pour {dataset_split}...")
            
            # 1. Accès direct à la table Arrow sous-jacente (.data)
            # Cela évite la conversion coûteuse Arrow -> Python List
            arrow_table = tokenized_ds[dataset_split].data
            
            # 2. On récupère la colonne "input_ids" qui est un ChunkedArray de Listes
            chunked_column = arrow_table.column("input_ids")
            
            # 3. Flattening C++ :
            # combine_chunks() fusionne les blocs mémoire
            # flatten() transforme List[List[int]] en List[int]
            # to_numpy() convertit en array numpy très rapidement
            flat_numpy = chunked_column.combine_chunks().flatten().to_numpy()
            
            # 4. Conversion finale en Tensor (très rapide depuis numpy)
            # On utilise int32 ou int64 selon la taille du vocabulaire, 
            # mais GPT2 utilise souvent int64 par défaut dans PyTorch.
            return torch.from_numpy(flat_numpy).to(dtype=torch.long)

    # Étape 2 : Création des tenseurs géants
    train_ids = flatten_to_tensor("train")
    print(f"[INFO] Nombre total de tokens train : {train_ids.numel():,}")

    val_ids = flatten_to_tensor("validation")
    print(f"[INFO] Nombre total de tokens val   : {val_ids.numel():,}")

    # Étape 3 : Création des Datasets
    train_dataset = TokenBlockDataset(train_ids, cfg.block_size)
    val_dataset = TokenBlockDataset(val_ids, cfg.block_size)

    print(f"[INFO] Nombre de séquences train : {len(train_dataset)}")
    print(f"[INFO] Nombre de séquences val   : {len(val_dataset)}")

    tokenization_time = time.time() - start_tokenization
    print(f"[INFO] Temps de tokenization total : {tokenization_time:.2f}s ({train_ids.numel() / tokenization_time:.0f} tokens/s)")

    return train_dataset, val_dataset