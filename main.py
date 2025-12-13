import time
import torch
from transformers import GPT2TokenizerFast

from decoderModel import DecoderModel  # suppose accessible dans le PYTHONPATH

def main():
    # ---------------- Hyperparams (ajustez au besoin) ----------------
    vocab_size = 50257  # GPT-2 BPE
    emb_dim = 512
    num_layers = 4
    num_groups = 4         # GQA: nb de groupes
    heads_per_group = 1    # = MHA si num_groups == total_heads
    active_experts = 2     # top-k
    total_experts = 4
    max_seq_len = 128

    batch_size = 4
    gen_len = 128           # longueur totale souhaitée (prompt inclus)
    temperature = 1.0
    top_k = 50
    top_p = 0.95
    use_kv_cache = True

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

    # ---------------- Prompts batch ----------------
    prompts = [
        "The quick brown fox",
        "In a distant future, humanity",
        "La recherche en intelligence artificielle",
        "Once upon a time",
    ]
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

    # ---------------- Mesure perf ----------------
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        out = model.sample(
            idx=idx,
            seq_len=min(gen_len, max_seq_len),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_kv_cache=use_kv_cache,
            eos_token_id=eos_id,
        )

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - start

    # tokens générés (hors prompt) ≈ batch_size * (gen_len - prompt_len_moy)
    gen_tokens_per_seq = out.size(1) - idx.size(1)
    total_new_tokens = batch_size * gen_tokens_per_seq
    tps = total_new_tokens / elapsed if elapsed > 0 else float("nan")

    print(f"Generated shape: {out.shape}")
    print(f"Elapsed: {elapsed:.3f}s, new tokens: {total_new_tokens}, tokens/s: {tps:.1f}")

    # ---------------- Décodage exemples ----------------
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    for i, txt in enumerate(texts):
        print(f"\n=== Sample {i} ===\n{txt}")

if __name__ == "__main__":
    main()