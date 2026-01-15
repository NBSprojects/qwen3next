# utils.py
import numpy as np
import torch
import torch.nn as nn
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def compute_layer_grad_norms(model: nn.Module) -> dict:
    """
    Calcule la norme L2 moyenne des gradients par composant de chaque layer.
    Retourne un dict {layer_idx: {"attn": norm, "ffn": norm, "norm": norm, "total": norm}}.
    """
    # Handle torch.compile wrapper
    layers = getattr(model, '_orig_mod', model).layers
    
    grad_norms = {}
    for layer_idx, layer in enumerate(layers):
        component_grads = {"attn": [], "ffn": [], "norm": [], "total": []}
        
        for name, param in layer.named_parameters():
            if param.grad is None:
                continue
            norm_val = param.grad.detach().norm().item()
            component_grads["total"].append(norm_val)
            
            # Catégorisation par composant
            if "gqa" in name:
                component_grads["attn"].append(norm_val)
            elif "moe" in name or "ffn" in name:
                component_grads["ffn"].append(norm_val)
            elif "norm" in name:
                component_grads["norm"].append(norm_val)
        
        grad_norms[layer_idx] = {
            k: sum(v) / len(v) if v else 0.0 
            for k, v in component_grads.items()
        }
    return grad_norms

def plot_training_curves(
    train_losses: list,
    eval_losses: list,
    save_path: str = "analytics/training_curves.png",
    use_moe: bool = True,
    dpi: int = 150,
):
    """
    Trace les courbes de loss (train et eval) et sauvegarde l'image.
    
    Args:
        train_losses: Liste de tuples (step, loss_value)
        eval_losses: Liste de tuples (step, loss_value)
        save_path: Chemin de sauvegarde du fichier PNG
        use_moe: True si modèle MoE, False si Dense (pour le titre)
        dpi: Résolution de l'image
    
    Returns:
        True si le plot a été généré, False sinon
    """
    if not train_losses and not eval_losses:
        print("[WARN] Aucune donnée de loss à tracer.")
        return False
    
    plt.figure(figsize=(10, 6))
    
    if train_losses:
        train_steps, train_vals = zip(*train_losses)
        plt.plot(train_steps, train_vals, label="Train Loss (CE)", color="blue", alpha=0.7)
    
    if eval_losses:
        eval_steps, eval_vals = zip(*eval_losses)
        plt.plot(eval_steps, eval_vals, label="Eval Loss", color="red", marker="o", linewidth=2)
    
    plt.xlabel("Step")
    plt.ylabel("Loss (Cross-Entropy)")
    model_type = "MoE" if use_moe else "Dense"
    plt.title(f"Training Curves - {model_type} Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Courbes de loss sauvegardées dans {save_path}")
    return True


def collect_layer_grad_norms(model: nn.Module, step: int, history: dict):
    """
    Collecte les normes moyennes des gradients par layer et les stocke dans history.
    
    Args:
        model: Le modèle (potentiellement compilé avec torch.compile)
        step: Le step actuel
        history: Dict mutable {layer_idx: {"attn": [], "ffn": [], "steps": []}}
    """
    grad_norms = compute_layer_grad_norms(model)
    for layer_idx, norms in grad_norms.items():
        if layer_idx not in history:
            history[layer_idx] = {"attn": [], "ffn": [], "steps": []}
        history[layer_idx]["attn"].append(norms["attn"])
        history[layer_idx]["ffn"].append(norms["ffn"])
        history[layer_idx]["steps"].append(step)


def plot_gradient_norms(
    grad_norm_history: dict,
    save_path: str = "analytics/gradient_norms.png",
    use_moe: bool = True,
    dpi: int = 150,
):
    """
    Trace les courbes de gradient norms par layer (Attention et FFN/MoE) et sauvegarde l'image.
    
    Args:
        grad_norm_history: Dict {layer_idx: {"attn": [], "ffn": [], "steps": []}}
        save_path: Chemin de sauvegarde du fichier PNG
        use_moe: True si modèle MoE, False si Dense (pour le titre)
        dpi: Résolution de l'image
    
    Returns:
        True si le plot a été généré, False sinon
    """
    if not grad_norm_history:
        print("[WARN] Aucune donnée de gradient norm à tracer.")
        return False
    
    n_layers = len(grad_norm_history)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Palette de couleurs distinctes pour chaque layer
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_layers))
    
    # Plot Attention grad norms
    ax_attn = axes[0]
    for layer_idx in sorted(grad_norm_history.keys()):
        data = grad_norm_history[layer_idx]
        ax_attn.plot(
            data["steps"], data["attn"],
            label=f"Layer {layer_idx}",
            color=colors[layer_idx],
            alpha=0.8,
            linewidth=1.2
        )
    ax_attn.set_xlabel("Step")
    ax_attn.set_ylabel("Gradient Norm (L2)")
    ax_attn.set_title("Attention Gradient Norms")
    ax_attn.legend(loc="upper right", fontsize=8)
    ax_attn.grid(True, alpha=0.3)
    ax_attn.set_yscale("log")
    
    # Plot FFN/MoE grad norms
    ax_ffn = axes[1]
    for layer_idx in sorted(grad_norm_history.keys()):
        data = grad_norm_history[layer_idx]
        ax_ffn.plot(
            data["steps"], data["ffn"],
            label=f"Layer {layer_idx}",
            color=colors[layer_idx],
            alpha=0.8,
            linewidth=1.2
        )
    ax_ffn.set_xlabel("Step")
    ax_ffn.set_ylabel("Gradient Norm (L2)")
    ax_ffn.set_title("FFN/MoE Gradient Norms")
    ax_ffn.legend(loc="upper right", fontsize=8)
    ax_ffn.grid(True, alpha=0.3)
    ax_ffn.set_yscale("log")
    
    model_type = "MoE" if use_moe else "Dense"
    fig.suptitle(f"Gradient Norms per Layer - {model_type} Model", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Courbes de gradient norms sauvegardées dans {save_path}")
    return True

# ------------------------------------------------------------------- #
#  ACTIVATION STATS (mean/std) — safe w/ torch.compile
# ------------------------------------------------------------------- #

@torch.no_grad()
def compute_layer_activation_stats(
    model: nn.Module,
    input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule (mean, std) des activations *en sortie de chaque layer*.

    Important pour le throughput avec torch.compile :
    - on n'utilise PAS de hooks (souvent source de graph breaks)
    - on passe par le modèle d'origine (model._orig_mod) si le modèle est compilé
    - on ne fait qu'un seul transfert CPU à la fin (pas de .item() dans la boucle)

    Args:
        model: modèle (potentiellement torch.compile)
        input_ids: LongTensor [B, L] sur le device

    Returns:
        means: Tensor [n_layers] (float32)
        stds : Tensor [n_layers] (float32)
    """

    # Handle torch.compile wrapper
    m = getattr(model, "_orig_mod", model)

    # Forward manuel (équivalent à model.forward avec use_cache=False)
    x = m.tok_emb(input_ids)

    means = []
    stds = []
    for layer in m.layers:
        x, _ = layer(x, kv_cache=None, use_cache=False)

        # Varmean en 1 passe. correction=0 => variance population (pas "unbiased").
        var, mean = torch.var_mean(x.to(torch.float32), dim=None, correction=0)
        means.append(mean)
        stds.append(torch.sqrt(var))

    return torch.stack(means, dim=0), torch.stack(stds, dim=0)


def collect_layer_activation_stats(
    model: nn.Module,
    input_ids: torch.Tensor,
    step: int,
    history: dict
):
    """
    Collecte mean/std des activations par layer et les stocke dans history.

    Structure :
        history[layer_idx] = {"mean": [...], "std": [...], "steps": [...]}

    Note perf : cette collecte fait un forward additionnel en no_grad,
    uniquement aux steps demandés (cf. cfg.activation_log_interval).
    """

    means, stds = compute_layer_activation_stats(model, input_ids)

    # 1 seul transfert CPU (vecteurs de taille n_layers)
    means_list = means.detach().cpu().tolist()
    stds_list = stds.detach().cpu().tolist()

    for layer_idx, (mval, sval) in enumerate(zip(means_list, stds_list)):
        if layer_idx not in history:
            history[layer_idx] = {"mean": [], "std": [], "steps": []}
        history[layer_idx]["mean"].append(float(mval))
        history[layer_idx]["std"].append(float(sval))
        history[layer_idx]["steps"].append(step)


def plot_activation_stats(
    activation_history: dict,
    save_path: str = "analytics/activation_stats.png",
    use_moe: bool = True,
    dpi: int = 150,
):
    """
    Trace les courbes mean/std des activations par layer et sauvegarde l'image.

    Args:
        activation_history: Dict {layer_idx: {"mean": [], "std": [], "steps": []}}
        save_path: Chemin de sauvegarde du fichier PNG
        use_moe: True si modèle MoE, False si Dense (pour le titre)
        dpi: Résolution de l'image

    Returns:
        True si le plot a été généré, False sinon
    """

    if not activation_history:
        print("[WARN] Aucune donnée d'activation stats à tracer.")
        return False

    n_layers = len(activation_history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_layers))

    # Mean
    ax_mean = axes[0]
    for layer_idx in sorted(activation_history.keys()):
        data = activation_history[layer_idx]
        ax_mean.plot(
            data["steps"],
            data["mean"],
            label=f"Layer {layer_idx}",
            color=colors[layer_idx],
            alpha=0.8,
            linewidth=1.2,
        )
    ax_mean.set_xlabel("Step")
    ax_mean.set_ylabel("Mean(x)")
    ax_mean.set_title("Activation Mean per Layer")
    ax_mean.legend(loc="upper right", fontsize=8)
    ax_mean.grid(True, alpha=0.3)

    # Std
    ax_std = axes[1]
    for layer_idx in sorted(activation_history.keys()):
        data = activation_history[layer_idx]
        ax_std.plot(
            data["steps"],
            data["std"],
            label=f"Layer {layer_idx}",
            color=colors[layer_idx],
            alpha=0.8,
            linewidth=1.2,
        )
    ax_std.set_xlabel("Step")
    ax_std.set_ylabel("Std(x)")
    ax_std.set_title("Activation Std per Layer")
    ax_std.legend(loc="upper right", fontsize=8)
    ax_std.grid(True, alpha=0.3)

    model_type = "MoE" if use_moe else "Dense"
    fig.suptitle(f"Activation Stats per Layer - {model_type} Model", fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Activation stats sauvegardées dans {save_path}")
    return True