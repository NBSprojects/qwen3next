import time
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")  # backend sans affichage
import matplotlib.pyplot as plt

from model import DecoderOnlyLM

from configs import SampleConfig


# ------------------------------------------------------------------- #
#  CONFIG
# ------------------------------------------------------------------- #
'''
@dataclass
class BenchmarkConfig:
    # --- modèle ---
    n_layers: int = 4
    emb_dim: int = 512
    num_groups: int = 4
    heads_per_group: int = 4
    active_experts: int = 2
    total_experts: int = 8
    tie_embeddings: bool = True

    # tokenizer GPT-2
    gpt2_model_name: str = "gpt2"

    # --- sampling ---
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    use_kv_cache: bool = True

    # prompt
    prompt: str = "Once upon a time"

    # checkpoints de mesure : tous les 64 tokens
    measure_every: int = 64

    # ckpt
    load_ckpt: bool = True
    ckpt_path: str = "model_train.ckpt"

    # device: "cuda", "cpu" ou "auto"
    device: str = "cuda"
'''

# ------------------------------------------------------------------- #
#  UTILS
# ------------------------------------------------------------------- #

def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def select_device(cfg: SampleConfig) -> torch.device:
    if cfg.device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise RuntimeError("CUDA demandé mais non disponible.")
    if cfg.device == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype_for_device(device: torch.device) -> torch.dtype:
    # Si GPU supporte bf16, on l'utilise. Sinon float32.
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # fallback : float16 possible, mais tu as demandé bf16 -> on reste en float32 pour la simplicité
        return torch.float32
    # Sur CPU, bf16 n'est pas toujours bien supporté pour tous les ops -> on reste aussi en float32
    return torch.float32


# ------------------------------------------------------------------- #
#  SAMPLING AVEC MESURE DE TEMPS
# ------------------------------------------------------------------- #

def generate_with_timing(
    model: DecoderOnlyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_kv_cache: bool = True,
    eos_token_id: Optional[int] = None,
    measure_points: Optional[List[int]] = None,
):

    """
    Version custom de sampling, très proche de DecoderOnlyLM.sample,
    mais avec instrumentation pour mesurer le temps de génération
    à différents nombres de tokens générés.

    Retourne :
      - generated_ids : [B, L_total]
      - timings       : dict {n_tokens_generated: temps_cumulé_en_secondes}
      - total_time    : temps total (s)
      - total_new     : nb total de tokens générés
    """
    model.eval()
    device = input_ids.device

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, L]

    generated = input_ids
    kv_cache = None
    new_tokens = 0

    if measure_points is None:
        measure_points = []
    measure_points = sorted(set([m for m in measure_points if m > 0]))

    timings = {m: None for m in measure_points}
    next_points_iter = iter(measure_points)
    next_point = next(next_points_iter, None)

    # --- timers ---
    use_cuda_timing = device.type == "cuda"

    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
    else:
        start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if use_kv_cache:
                if kv_cache is None:
                    # premier passage : tout le prompt
                    logits, _, kv_cache = model(
                        generated, kv_cache=None, use_cache=True
                    )
                else:
                    # passes suivantes : uniquement le dernier token
                    logits, _, kv_cache = model(
                        generated[:, -1:], kv_cache=kv_cache, use_cache=True
                    )
            else:
                # pas de cache : on repasse tout le contexte
                logits, _ = model(generated, kv_cache=None, use_cache=False)

            logits_last = logits[:, -1, :]  # [B, vocab]

            # on réutilise le staticmethod déjà codé dans DecoderOnlyLM
            next_token = model._sample_from_logits(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            new_tokens += 1

            # mesure à certains checkpoints : 64, 128, 192, 256, ...
            if next_point is not None and new_tokens == next_point:
                if use_cuda_timing:
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed_ms = starter.elapsed_time(ender)  # depuis le début
                    timings[next_point] = elapsed_ms / 1000.0
                else:
                    now = time.perf_counter()
                    timings[next_point] = now - start_time
                next_point = next(next_points_iter, None)

            if eos_token_id is not None:
                if (generated[:, -1] == eos_token_id).all():
                    break

    # temps total
    if use_cuda_timing:
        ender.record()
        torch.cuda.synchronize()
        total_time_sec = starter.elapsed_time(ender) / 1000.0
    else:
        total_time_sec = time.perf_counter() - start_time

    return generated, timings, total_time_sec, new_tokens


# ------------------------------------------------------------------- #
#  MAIN
# ------------------------------------------------------------------- #

def main():
    cfg = SampleConfig()

    # --- Device & dtype ---
    device = select_device(cfg)
    dtype = get_dtype_for_device(device)

    print(f"[INFO] Device : {device}")
    print(f"[INFO] Dtype  : {dtype}")

    # --- Tokenizer GPT-2 BPE ---
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ImportError("Veuillez installer transformers: pip install transformers")

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.gpt2_model_name)
    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    print(f"[INFO] GPT-2 vocab size : {vocab_size}")
    print(f"[INFO] EOS token id    : {eos_token_id}")

    # --- Instanciation du modèle ---
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        emb_dim=cfg.emb_dim,
        n_layers=cfg.n_layers,
        num_groups=cfg.num_groups,
        heads_per_group=cfg.heads_per_group,
        active_experts=cfg.active_experts,
        total_experts=cfg.total_experts,
        tie_embeddings=cfg.tie_embeddings,
    )

    # --- Chargement éventuel d'un checkpoint ---
    if cfg.load_ckpt:
        print(f"[INFO] Chargement du checkpoint depuis {cfg.ckpt_path}")
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        
        # Clé correcte
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        
        # Retirer le préfixe _orig_mod. ajouté par torch.compile()
        new_state = {}
        for k, v in state.items():
            if k.startswith("_orig_mod."):
                new_state[k[len("_orig_mod."):]] = v
            else:
                new_state[k] = v
        
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[INFO] Keys manquantes : {len(missing)} ; keys inattendues : {len(unexpected)}")

    # --- Comptage des paramètres ---
    print("\n[PARAMÈTRES ENTRAÎNABLES PAR BLOC]")

    emb_params = count_trainable_params(model.tok_emb)
    print(f"Embedding      : {emb_params:,} params")

    for i, layer in enumerate(model.layers):
        layer_params = count_trainable_params(layer)
        print(f"DecoderGQA {i:02d} : {layer_params:,} params")

    norm_out_params = count_trainable_params(model.norm_out)
    lm_head_params = count_trainable_params(model.lm_head)
    print(f"Norm finale    : {norm_out_params:,} params")
    print(f"LM head        : {lm_head_params:,} params")

    total_params = count_trainable_params(model)
    print(f"TOTAL          : {total_params:,} params "
          f"({total_params / 1e6:.2f} M)\n")

    # --- Conversion en dtype (bf16 si possible) + device ---
    model = model.to(device=device)
    if dtype == torch.bfloat16:
        print("[INFO] Conversion du modèle en bfloat16...")
        model = model.to(dtype=torch.bfloat16)
    else:
        print("[INFO] BF16 non utilisé (fallback float32).")


    # --- Préparation du prompt ---
    enc = tokenizer(
        cfg.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)

    print(f"[INFO] Prompt : {cfg.prompt!r}")

    # --- Points de mesure : 64, 128, 192, ... <= max_new_tokens ---
    measure_points = list(
        range(cfg.measure_every, cfg.max_new_tokens + 1, cfg.measure_every)
    )

    # --- Génération avec mesure de temps ---
    generated_ids, timings, total_time, total_new = generate_with_timing(
        model=model,
        input_ids=input_ids,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        use_kv_cache=cfg.use_kv_cache,
        eos_token_id=eos_token_id,
        measure_points=measure_points,
    )

    # --- Décodage du texte (optionnel, juste pour vérif) ---
    try:
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n[GENERATION SAMPLE]")
        print(gen_text)
        print()
    except Exception as e:
        print(f"[WARN] Erreur lors du decode du texte : {e}")

    print(f"\n[TOTAL]")
    print(f"  Tokens générés : {total_new}")
    print(f"  Temps total     : {total_time:.4f} s")
    if total_time > 0:
        print(f"  Tokens / seconde: {total_new / total_time:.2f}")

    # --- Graphique per_token_times.png ---
    xs = []
    ys = []
    for m in measure_points:
        if timings[m] is not None:
            xs.append(m)
            ys.append(timings[m])

    if xs:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Nombre de tokens générés")
        plt.ylabel("Temps cumulé (s)")
        plt.title(f"Latence de génération (use_kv_cache={cfg.use_kv_cache})")
        plt.grid(True)
        plt.tight_layout()
        out_path = "analytics/per_token_times.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\n[INFO] Graphique sauvegardé dans {out_path}")
    else:
        print("\n[WARN] Aucun point de mesure valide pour tracer la courbe.")


if __name__ == "__main__":
    main()
