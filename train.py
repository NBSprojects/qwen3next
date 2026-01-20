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

from transformers import GPT2TokenizerFast

from model import DecoderOnlyLM, DecoderOnlyLMDense
from utils import (
    collect_layer_grad_norms,
    collect_layer_activation_stats,
    plot_gradient_norms,
    plot_training_curves,
    plot_activation_stats,
    set_seed,
    select_device,
    count_trainable_params,
    param_groups_for_wd,
    get_avg_tokens_per_expert,
    kl_to_uniform_from_counts,
    plot_moe_kl_curve,
)
from dataprep import build_token_datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import TrainConfig



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

    for layer in model.layers:
        layer.norm1 = layer.norm1.float()
        layer.norm2 = layer.norm2.float()
        layer.moe.gate_layer = layer.moe.gate_layer.float()

    model.norm_out = model.norm_out.float()

    # torch.compile après .to(...)
    print("[INFO] Compilation du modèle avec torch.compile...")
    model = torch.compile(model, mode="max-autotune")

    # Optimiseur AdamW (classique, weight decay)
    # On essaye fused=True si dispo
    try:
        optimizer = torch.optim.AdamW(
            param_groups_for_wd(model, cfg.weight_decay),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            fused=True,  
        )
        print("[INFO] AdamW(fused=True)")
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups_for_wd(model, cfg.weight_decay),
            lr=cfg.learning_rate,
            betas=cfg.betas,
        )
        print("[INFO] AdamW(fused=False)")


    # ------------------------------------------------------------------ #
    # Scheduler OneCycleLR
    # ------------------------------------------------------------------ #
    scheduler = None
    if cfg.use_onecycle_scheduler:
        # Calcul de final_div_factor à partir de min_lr
        # OneCycleLR: final_lr = initial_lr / final_div_factor
        #             initial_lr = max_lr / div_factor
        # Donc: final_lr = max_lr / div_factor / final_div_factor
        # => final_div_factor = (max_lr / div_factor) / min_lr
        initial_lr = cfg.learning_rate / cfg.div_factor
        final_div_factor = initial_lr / cfg.min_lr if cfg.min_lr > 0 else 1e4
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=cfg.max_steps,
            pct_start=cfg.pct_start,
            anneal_strategy=cfg.anneal_strategy,
            div_factor=cfg.div_factor,
            final_div_factor=final_div_factor,
        )
        print(f"[INFO] OneCycleLR: max_lr={cfg.learning_rate:.2e}, "
              f"pct_start={cfg.pct_start}, div_factor={cfg.div_factor}, "
              f"final_div_factor={final_div_factor:.1f}, anneal={cfg.anneal_strategy}")


    # ------------------------------------------------------------------ #
    # Boucle de training
    # ------------------------------------------------------------------ #

    model.train()

    # Itérateur infini sur le DataLoader pour max_steps
    train_iter = iter(train_loader)

    global_step = 0

    train_losses = []
    eval_losses = []
    grad_norm_history = {}  # {layer_idx: {"attn": [], "ffn": [], "steps": []}}
    activation_history = {}  # {layer_idx: {"mean": [], "std": [], "steps": []}}
    moe_kl_history = []

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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient monitoring
        if cfg.grad_log_interval is not None and global_step % cfg.grad_log_interval == 0:
            collect_layer_grad_norms(model, global_step, grad_norm_history)

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


        # Logging
        if global_step % cfg.log_interval == 0:
            if cfg.use_moe:
                lb_val = float(lb_loss.detach().cpu())
            else:
                lb_val = 0.0

            loss_ce_val = float(loss_ce.detach().cpu())
            train_losses.append((global_step, loss_ce_val))
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[step {global_step:05d}] "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"lr={current_lr:.2e}"
                f"(ce={float(loss_ce.detach().cpu()):.4f}, lb={lb_val:.4f}) "
                f"| steps/s={steps_per_sec:.2f}"
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

        # Activation monitoring (mean/std par layer)
        if ( cfg.activation_log_interval is not None and global_step % cfg.activation_log_interval == 0):
            collect_layer_activation_stats(model, input_ids, global_step, activation_history)

        # MoE KL monitoring (token distribution vs uniform)
        if (cfg.use_moe 
            and cfg.moe_kl_log_interval is not None 
            and global_step % cfg.moe_kl_log_interval == 0):
            avg_counts = get_avg_tokens_per_expert(model)
            if avg_counts is not None:
                kl_val = kl_to_uniform_from_counts(avg_counts)
                moe_kl_history.append((global_step, kl_val.item()))

        if global_step >= cfg.max_steps:
            break

    # ------------------------------------------------------------------ #
    # Sauvegarde du checkpoint
    # ------------------------------------------------------------------ #

    if(cfg.save_model):
        ckpt = {
            "config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
            "tokenizer_name": cfg.gpt2_model_name,
            "vocab_size": vocab_size,
            "use_moe": cfg.use_moe,
        }
        torch.save(ckpt, cfg.save_ckpt_path)
        print(f"[INFO] Checkpoint sauvegardé dans {cfg.save_ckpt_path}")
    else:
        print(f"[INFO] Checkpoint non sauvegardé")

    # Plotting des courbes de loss
    plot_training_curves(
        train_losses,
        eval_losses,
        save_path="analytics/training_curves.png",
        use_moe=cfg.use_moe,
    )
    
    # Plotting des gradient norms
    plot_gradient_norms(grad_norm_history, save_path="analytics/gradient_norms.png", use_moe=cfg.use_moe)
    
    # Plotting des activation stats
    plot_activation_stats(activation_history, save_path="analytics/activation_stats.png", use_moe=cfg.use_moe)

    # Plotting des MoE KL stats (distribution tokens/expert vs uniforme)
    if cfg.use_moe:
        plot_moe_kl_curve("analytics", moe_kl_history)




if __name__ == "__main__":
    main()
