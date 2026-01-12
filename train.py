# train.py

import math
import os
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import GPT2TokenizerFast

from model import DecoderOnlyLM, DecoderOnlyLMDense

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------- #
#  CONFIG
# ------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    # --- Modèle (identique à benchmark_sampling pour le MoE) ---
    n_layers: int = 4
    emb_dim: int = 512
    num_groups: int = 4
    heads_per_group: int = 4
    active_experts: int = 2
    total_experts: int = 8
    tie_embeddings: bool = True
    use_moe: bool = True  # True = GQA+MoE, False = GQA+Dense

    # --- Données ---
    gpt2_model_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"

    block_size: int = 256  # context length FIXE
    batch_size: int = 32
    num_workers: int = 4

    # --- Training ---
    max_steps: int = 10_000
    eval_interval: int = 200
    log_interval: int = 20

    learning_rate: float = 3e-4
    min_lr: float = 1e-5  
    use_cosine_scheduler: bool = True
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: Optional[float] = 1.0
    lambda_moe: float = 0.01  # coefficient pour le balancing loss

    seed: int = 147

    # --- Device ---
    device: str = "cuda"  # "cuda", "cpu" ou "auto"

    # --- Checkpoint ---
    save_ckpt_path: str = "model_train.ckpt"


# ------------------------------------------------------------------- #
#  UTILS
# ------------------------------------------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA demandé mais non disponible.")
    if device_str == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------- #
#  DATASET : fixed-length token blocks
# ------------------------------------------------------------------- #

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




# ------------------------------------------------------------------- #
#  EVALUATION
# ------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, cfg: TrainConfig):
    """
    Évalue la perplexité (loss) sur une partie du val set.
    """
    model.eval()
    losses = []

    for batch in val_loader:
        # batch: [B, block_size]
        batch = batch.to(device)
        # On prédit le token suivant : input =[:-1], target = [1:]
        input_ids = batch[:, :-1]  # [B, block_size-1]
        targets = batch[:, 1:]     # [B, block_size-1]

        logits, lb_loss = model(input_ids)  # use_cache=False par défaut
        # On calcule la CE en float32 pour la stabilité
        logits_f32 = logits.to(torch.float32)

        loss_ce = F.cross_entropy(
            logits_f32.reshape(-1, logits_f32.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        losses.append(loss_ce.item())

    model.train()
    return float(np.mean(losses))


# ------------------------------------------------------------------- #
#  MAIN TRAIN LOOP
# ------------------------------------------------------------------- #

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = select_device(cfg.device)
    print(f"[INFO] Device : {device}")

    # On impose bfloat16 pour le modèle (params + activations)
    dtype = torch.bfloat16
    print(f"[INFO] Dtype  : {dtype} (tout le modèle tourne en bf16)")

    # Tokenizer GPT-2 (identique pour tous les runs)
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.gpt2_model_name)
    vocab_size = tokenizer.vocab_size
    print(f"[INFO] GPT-2 vocab size : {vocab_size}")

    # Datasets + DataLoaders (bloc de taille fixe 256)
    train_dataset, val_dataset = build_token_datasets(cfg, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,  # pour garder [B, block_size] constant
        num_workers=cfg.num_workers,
        pin_memory=True,              
        persistent_workers=True    
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,               # Prépare la RAM pour un transfert rapide vers le GPU
        persistent_workers=True    
    )

    # ------------------------------------------------------------------ #
    # Construction du modèle (Dense ou MoE)
    # ------------------------------------------------------------------ #
    if cfg.use_moe:
        print("[INFO] Instanciation du modèle GQA + MoE")
        model: nn.Module = DecoderOnlyLM(
            vocab_size=vocab_size,
            emb_dim=cfg.emb_dim,
            n_layers=cfg.n_layers,
            num_groups=cfg.num_groups,
            heads_per_group=cfg.heads_per_group,
            active_experts=cfg.active_experts,
            total_experts=cfg.total_experts,
            tie_embeddings=cfg.tie_embeddings,
        )
    else:
        print("[INFO] Instanciation du modèle GQA + Dense FFN")
        model = DecoderOnlyLMDense(
            vocab_size=vocab_size,
            emb_dim=cfg.emb_dim,
            n_layers=cfg.n_layers,
            num_groups=cfg.num_groups,
            heads_per_group=cfg.heads_per_group,
            tie_embeddings=cfg.tie_embeddings,
        )

    # Nombre de params
    total_params = count_trainable_params(model)
    print(f"[INFO] Nombre total de paramètres entraînables : {total_params:,} "
          f"({total_params / 1e6:.2f} M)")

    # Pour le MoE, on fixe la capacité par expert pour avoir des shapes 100% stables
    if cfg.use_moe:
        tokens_per_step = cfg.batch_size * (cfg.block_size - 1)  # seq_len=block_size-1
        sample_moe = model.layers[0].moe
        cap = sample_moe._compute_capacity(tokens_per_step)
        print(f"[INFO] Capacité MoE fixée à {cap} slots / expert")
        for layer in model.layers:
            layer.moe.capacity = cap  # override => plus dépendant du batch

    # Move to device + bf16
    model = model.to(device=device)
    model = model.to(dtype=dtype)

    # torch.compile après .to(...)
    print("[INFO] Compilation du modèle avec torch.compile...")
    model = torch.compile(model, mode="max-autotune")

    # Optimiseur AdamW (classique, weight decay)
    # On essaye fused=True si dispo
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        print("[INFO] AdamW(fused=True)")
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        print("[INFO] AdamW(fused=False)")


    scheduler = None
    if cfg.use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_steps,
            eta_min=cfg.min_lr,
        )


    # ------------------------------------------------------------------ #
    # Boucle de training
    # ------------------------------------------------------------------ #

    model.train()

    # Itérateur infini sur le DataLoader pour max_steps
    train_iter = iter(train_loader)

    global_step = 0

    train_losses = []
    eval_losses = []

    while global_step < cfg.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        global_step += 1

        # batch : [B, block_size]
        batch = batch.to(device, non_blocking=True)

        # input_ids pour le modèle : tous sauf le dernier token
        input_ids = batch[:, :-1]  # [B, block_size-1]
        targets = batch[:, 1:]     # [B, block_size-1]

        # Mesure du temps par step (forward+backward+update)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        # Forward (sans KV cache, use_cache=False par défaut)
        logits, lb_loss = model(input_ids)

        # Loss CE en float32 + balancing MoE si applicable
        logits_f32 = logits.to(torch.float32)
        loss_ce = F.cross_entropy(
            logits_f32.reshape(-1, logits_f32.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )

        if cfg.use_moe:
            loss = loss_ce + cfg.lambda_moe * lb_loss.to(torch.float32)
        else:
            loss = loss_ce

        optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_end = time.perf_counter()
        step_time = step_end - step_start
        steps_per_sec = 1.0 / step_time if step_time > 0 else float("inf")

        # Stats MoE : tokens moyens par expert
        tokens_per_expert_str = ""
        if cfg.use_moe:
            tokens_per_expert_layers = []
            for layer in model.layers:
                moe = getattr(layer, "moe", None)
                if moe is not None and moe.last_tokens_per_expert is not None:
                    # last_tokens_per_expert : [num_exp], float32, no_grad
                    tokens_per_expert_layers.append(moe.last_tokens_per_expert.detach())
            if tokens_per_expert_layers:
                stacked = torch.stack(tokens_per_expert_layers, dim=0)  # [n_layers, num_exp]
                avg_tokens_per_expert = stacked.mean(dim=0)             # [num_exp]
                tokens_per_expert_list = avg_tokens_per_expert.cpu().tolist()
                tokens_per_expert_str = " | tokens/expert=" + ", ".join(
                    f"{v:.1f}" for v in tokens_per_expert_list
                )

        # Logging
        if global_step % cfg.log_interval == 0:
            if cfg.use_moe:
                lb_val = float(lb_loss.detach().cpu())
            else:
                lb_val = 0.0

            loss_ce_val = float(loss_ce.detach().cpu())
            train_losses.append((global_step, loss_ce_val))

            print(
                f"[step {global_step:05d}] "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"(ce={float(loss_ce.detach().cpu()):.4f}, lb={lb_val:.4f}) "
                f"| steps/s={steps_per_sec:.2f}"
                f"{tokens_per_expert_str}"
            )

        # Éval régulière
        if global_step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, cfg)
            val_ppl = math.exp(val_loss)

            eval_losses.append((global_step, val_loss))

            print(
                f"[EVAL] step {global_step:05d} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}"
            )

        if global_step >= cfg.max_steps:
            break

    # ------------------------------------------------------------------ #
    # Sauvegarde du checkpoint
    # ------------------------------------------------------------------ #

    ckpt = {
        "config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
        "tokenizer_name": cfg.gpt2_model_name,
        "vocab_size": vocab_size,
        "use_moe": cfg.use_moe,
    }
    torch.save(ckpt, cfg.save_ckpt_path)
    print(f"[INFO] Checkpoint sauvegardé dans {cfg.save_ckpt_path}")

    # Plotting des courbes de loss
    if train_losses or eval_losses:
        plt.figure(figsize=(10, 6))
        
        if train_losses:
            train_steps, train_vals = zip(*train_losses)
            plt.plot(train_steps, train_vals, label="Train Loss (CE)", color="blue", alpha=0.7)
        
        if eval_losses:
            eval_steps, eval_vals = zip(*eval_losses)
            plt.plot(eval_steps, eval_vals, label="Eval Loss", color="red", marker="o", linewidth=2)
        
        plt.xlabel("Step")
        plt.ylabel("Loss (Cross-Entropy)")
        plt.title(f"Training Curves - {'MoE' if cfg.use_moe else 'Dense'} Model")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[INFO] Courbes de loss sauvegardées dans {plot_path}")


if __name__ == "__main__":
    main()
