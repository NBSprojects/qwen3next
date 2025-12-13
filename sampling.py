import time
import torch
from transformers import GPT2TokenizerFast
import argparse
import matplotlib.pyplot as plt

from decoderModel import DecoderModel  # suppose accessible dans le PYTHONPATH

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def main():

    parser = argparse.ArgumentParser()
    # Si on met --use_kvcache, ça devient True. Sinon False par défaut.
    parser.add_argument("--use_kvcache", action="store_true") 

    # Idem pour compile
    parser.add_argument("--use_compile", action="store_true")
    args = parser.parse_args()
    use_kv_cache = args.use_kvcache
    use_compile = args.use_compile
    print(f"use_kv_cache: {use_kv_cache}")
    print(f"use_compile: {use_compile}")

    # ---------------- Hyperparams (ajustez au besoin) ----------------
    vocab_size = 50257  # GPT-2 BPE
    emb_dim = 512
    num_layers = 4
    num_groups = 4         # GQA: nb de groupes
    heads_per_group = 4    # = MHA si num_groups == total_heads
    active_experts = 2     # top-k
    total_experts = 8
    max_seq_len = 2048

    gen_len = max_seq_len           # longueur totale souhaitée (prompt inclus)
    temperature = 1.0
    top_k = 50
    top_p = 0.95


    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # ---------------- Tokenizer ----------------
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # pour batcher proprement
    eos_id = tokenizer.eos_token_id

    # ---------------- Modèle ----------------
    torch.manual_seed(0)
    model = DecoderModel(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_groups=num_groups,
        heads_per_group=heads_per_group,
        active_experts=active_experts,
        total_experts=total_experts,
        max_seq_len=max_seq_len,
    ).to(device)
    model = model.to(dtype)
    model.eval()

    if use_compile: 
        model = torch.compile(
            model,
            mode="max-autotune",
            fullgraph=True
        )

    # ---------------- Comptage de paramètres ----------------
    emb_params = count_params(model.token_embedding)
    layer_params = [count_params(layer) for layer in model.layers]
    final_norm_params = count_params(model.final_norm)
    lm_head_params = count_params(model.lm_head)

    # lm_head est tied avec l'embedding : ne pas le recompter dans le total unique
    total_unique = emb_params + sum(layer_params) + final_norm_params

    print(f"Total params (uniques): {total_unique/1e6:.2f}M")
    print(f" - token embedding (shared with lm_head): {emb_params/1e6:.2f}M")
    for i, lp in enumerate(layer_params):
        print(f" - layer {i}: {lp/1e6:.2f}M")
    print(f" - final_norm: {final_norm_params/1e6:.4f}M")
    print(f" - lm_head (tied, non compté dans total): {lm_head_params/1e6:.2f}M")

    # ---------------- Prompts batch ----------------
    prompts = [
        "The quick brown fox",
    ]
    batch_size = len(prompts)


    prompts = prompts[:batch_size]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    idx = enc.input_ids.to(device)

    # ---------------- Warmup (optionnel) ----------------
    with torch.no_grad():
        _ = model.sample(
            idx=idx,
            seq_len=min(gen_len, max_seq_len),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_kv_cache=use_kv_cache,
            eos_token_id=eos_id,
        )

    # ---------------- Mesure cumulative (points tous les 64 tokens) ----------------
    B, T0 = idx.shape
    kv_caches = None
    generated = idx[:, 0:1]
    cur_tokens = idx[:, 0]
    total_steps = min(gen_len, max_seq_len) - 1

    # stocke des points (nb_tokens_generes, temps_cumule_s)
    cum_points = [(0, 0.0)]

    torch.cuda.synchronize() if device == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        for step in range(total_steps):
            logits, kv_caches, _ = model._decode_step(cur_tokens, kv_caches)

            # force le prompt si on est encore dedans
            if step + 1 < T0:
                next_token = idx[:, step + 1 : step + 2]
            else:
                next_token = model._sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            generated = torch.cat([generated, next_token], dim=1)
            cur_tokens = next_token.squeeze(-1)

            # tous les 64 tokens générés (par séquence) + fin
            tokens_generated_per_seq = generated.size(1) - T0
            if tokens_generated_per_seq % 64 == 0 or generated.size(1) >= gen_len:
                torch.cuda.synchronize() if device == "cuda" else None
                t_now = time.perf_counter() - t_start
                cum_points.append((tokens_generated_per_seq, t_now))

            if generated.size(1) >= gen_len:
                break

    out = generated
    # stats finales
    total_tokens = out.size(1) - T0
    total_time = cum_points[-1][1]
    total_new_tokens = B * total_tokens
    tps = total_new_tokens / total_time if total_time > 0 else float("nan")

    print(f"Generated shape: {out.shape}")
    print(f"Elapsed (cumulé): {total_time:.3f}s, new tokens: {total_new_tokens}, tokens/s: {tps:.1f}")

    # ---------------- Courbe cumulative ----------------
    xs, ys = zip(*cum_points)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Tokens générés par séquence")
    plt.ylabel("Temps cumulé (s)")
    plt.title("Temps cumulé vs tokens générés (points tous les 64 tokens)")
    plt.tight_layout()
    plt.savefig("per_token_times.png", dpi=150, bbox_inches="tight")

    # ---------------- Décodage exemples ----------------
    #texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    #for i, txt in enumerate(texts):
    #    print(f"\n=== Sample {i} ===\n{txt}")

if __name__ == "__main__":
    main()