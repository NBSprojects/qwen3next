# decoderModel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import DecoderGQALayer, RMSNorm


class DecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        num_layers: int,
        num_groups: int,
        heads_per_group: int,
        active_experts: int,
        total_experts: int,
        max_seq_len: int = 256,  # seq_size fixé à 256 par défaut
    ):
        super().__init__()

        # ⚠️ IMPORTANT : bug dans GroupRopeAttention => il ne fonctionne correctement
        # que pour num_heads == 1 (voir explication plus bas).
        if heads_per_group != 1:
            raise ValueError(
                "GroupRopeAttention tel qu'implémenté dans gqa.py ne supporte "
                "pas heads_per_group > 1 (problème de dimensions dans les einsum). "
                "Utilise heads_per_group=1 ou corrige GroupRopeAttention."
            )

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.heads_per_group = heads_per_group
        self.active_experts = active_experts
        self.total_experts = total_experts
        self.max_seq_len = max_seq_len

        # Embedding des tokens
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # Pile de couches DecoderGQALayer
        self.layers = nn.ModuleList(
            [
                DecoderGQALayer(
                    num_groups,
                    heads_per_group,
                    active_experts,
                    total_experts,
                    emb_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Norm finale avant la tête de langage
        self.final_norm = RMSNorm(emb_dim)

        # Projection vers les logits vocabulaire
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Weight tying Embedding <-> lm_head
        self.lm_head.weight = self.token_embedding.weight

    # ------------------------------------------------------------------
    # Forward "training" classique (pas de KV cache)
    # ------------------------------------------------------------------
    def forward(self, idx, return_load_balancing_loss: bool = False):
        """
        idx : LongTensor [batch_size, seq_len] (ids de tokens)

        Retourne :
          - logits : [batch_size, seq_len, vocab_size]
          - optionnellement : load_balancing_loss scalaire (Moyenne des pertes MoE)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds model max_seq_len {self.max_seq_len}"
        )

        # [B, T, emb_dim]
        x = self.token_embedding(idx)

        load_losses = []
        for layer in self.layers:
            # On ne passe pas de KV cache ici (use_cache=False)
            x, lb = layer(x, kv_cache=None, use_cache=False)
            load_losses.append(lb)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        if return_load_balancing_loss:
            load_balancing_loss = torch.stack(load_losses).mean()
            return logits, load_balancing_loss

        return logits

    # ------------------------------------------------------------------
    # Chemin "one step" avec KV caching (on réutilise les modules internes)
    # ------------------------------------------------------------------
    def _run_layer_step_with_cache(self, layer, x, layer_cache):
        """
        Applique UNE étape de la couche `layer` avec KV cache.

        layer       : DecoderGQALayer
        x           : [B, 1, emb_dim] (token courant uniquement)
        layer_cache : None ou liste[num_groups] de (K, V)

        Retourne :
          - x_out           : [B, 1, emb_dim]
          - new_layer_cache : même structure que layer_cache
          - lb              : load_balancing_loss de ce layer (scalaire)
        """
        x_in = x  # pour le résiduel

        # RMSNorm avant attention
        x = layer.norm1(x)

        G = layer.num_groups
        if layer_cache is None:
            layer_cache = [None] * G
        else:
            assert len(layer_cache) == G, "Bad kv cache for layer"

        attn_outputs = []
        new_layer_cache = []

        # On appelle directement les GroupRopeAttention internes
        for g_idx, gqa in enumerate(layer.gqas):
            out, kv = gqa(x, kv_cache=layer_cache[g_idx], use_cache=True)
            attn_outputs.append(out)
            new_layer_cache.append(kv)

        # Concat des groupes sur la dernière dimension
        attn_out = torch.cat(attn_outputs, dim=-1)  # [B, 1, emb_dim]

        # Résiduel attention
        x = x_in + attn_out
        x_mid = x

        # RMSNorm avant MoE
        x = layer.norm2(x)

        # MoE
        x_mlp, lb = layer.moe(x)  # [B, 1, emb_dim], scalaire

        # Résiduel MLP
        x_out = x_mid + x_mlp

        return x_out, new_layer_cache, lb

    def _decode_step(self, idx_t, kv_caches=None):
        """
        UNE étape autoregressive avec KV cache.

        idx_t     : [B] ids du token courant
        kv_caches : None ou liste[num_layers] de liste[num_groups] de (K, V)

        Retourne :
          - logits_t       : [B, vocab_size] logits pour le prochain token
          - new_kv_caches  : même structure que kv_caches
          - load_loss_step : scalaire (moyenne des pertes MoE sur les layers)
        """
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        else:
            assert len(kv_caches) == len(self.layers), "kv_caches bad length"

        # [B, 1, emb_dim]
        x = self.token_embedding(idx_t.unsqueeze(1))

        new_kv_caches = []
        load_losses = []

        for layer, layer_cache in zip(self.layers, kv_caches):
            x, new_layer_cache, lb = self._run_layer_step_with_cache(
                layer, x, layer_cache
            )
            new_kv_caches.append(new_layer_cache)
            load_losses.append(lb)

        # Norm + tête
        x = self.final_norm(x)            # [B, 1, emb_dim]
        logits = self.lm_head(x)          # [B, 1, vocab_size]
        logits_t = logits[:, -1, :]       # [B, vocab_size]

        load_balancing_loss = torch.stack(load_losses).mean()

        return logits_t, new_kv_caches, load_balancing_loss

    # ------------------------------------------------------------------
    # Utilitaires de sampling (top-k / top-p)
    # ------------------------------------------------------------------
    def _sample_from_logits(self, logits, temperature=1.0, top_k=None, top_p=None):
        """
        logits : [B, vocab_size]
        Retourne next_tokens : [B, 1]
        """
        if temperature is not None and temperature != 1.0:
            logits = logits / temperature

        # Top-k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k, dim=-1)
            min_values = values[..., -1, None]
            logits = torch.where(
                logits < min_values,
                logits.new_full(logits.shape, float("-inf")),
                logits,
            )

        # Top-p (nucleus)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1
            )
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            # Coupe au seuil top_p
            mask = cumulative_probs > top_p
            mask[..., 0] = False  # on garde toujours au moins 1 token

            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

            # On remet dans l'ordre original
            logits = logits.new_full(logits.shape, float("-inf"))
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]
        return next_tokens

    # ------------------------------------------------------------------
    # Sampling avec option KV cache
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        idx,
        seq_len: int = None,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        use_kv_cache: bool = True,
        eos_token_id: int = None,
    ):
        """
        Génère jusqu'à `seq_len` tokens au total (prompt compris).

        idx : [B, T0] tokens du prompt initial
        seq_len : longueur totale max (par défaut = self.max_seq_len)
        top_k, top_p : filtrage de la distribution
        use_kv_cache : True => on utilise le chemin incrémental avec caches KV
        eos_token_id : si non None, stoppe la génération par séquence à EOS
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        device = next(self.parameters()).device
        idx = idx.to(device)

        B, T0 = idx.shape
        if T0 == 0:
            raise ValueError("sample() expects at least one token in the prompt")

        if T0 >= seq_len:
            # Prompt déjà suffisant
            return idx[:, :seq_len]

        if not use_kv_cache:
            return self._sample_no_cache(
                idx, seq_len, temperature, top_k, top_p, eos_token_id
            )

        # --- Chemin avec KV cache ---
        # On "teacher force" tout le prompt, puis on échantillonne.
        generated = idx[:, 0:1].clone()  # on démarre avec le premier token
        kv_caches = None
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        total_steps = seq_len - 1  # on a déjà 1 token
        cur_tokens = idx[:, 0]     # token courant à l'étape 0

        for step in range(total_steps):
            logits, kv_caches, _ = self._decode_step(cur_tokens, kv_caches)

            if step + 1 < T0:
                # encore dans le prompt : on force le prochain token du prompt
                next_token = idx[:, step + 1 : step + 2]
            else:
                # on commence à sampler
                if eos_token_id is not None:
                    logits = logits.clone()
                    # pour les séquences déjà finies, on impose eos
                    logits[finished, :] = float("-inf")
                    logits[finished, eos_token_id] = 0.0

                next_token = self._sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                if eos_token_id is not None:
                    next_ids = next_token.squeeze(-1)
                    finished = finished | (next_ids == eos_token_id)

            generated = torch.cat([generated, next_token], dim=1)
            cur_tokens = next_token.squeeze(-1)

            if generated.size(1) >= seq_len:
                break

            if step + 1 >= T0 and eos_token_id is not None and finished.all():
                break

        return generated[:, :seq_len]

    # ------------------------------------------------------------------
    # Sampling sans KV cache (on re-forward toute la séquence à chaque step)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_no_cache(
        self,
        idx,
        seq_len,
        temperature,
        top_k,
        top_p,
        eos_token_id,
    ):
        device = next(self.parameters()).device
        generated = idx.to(device)

        B, T0 = generated.shape
        if T0 == 0:
            raise ValueError("sample() expects at least one token in the prompt")

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        while generated.size(1) < seq_len:
            logits, _ = self.forward(generated, return_load_balancing_loss=True)
            logits = logits[:, -1, :]  # logits du dernier token

            if eos_token_id is not None:
                logits = logits.clone()
                logits[finished, :] = float("-inf")
                logits[finished, eos_token_id] = 0.0

            next_token = self._sample_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if eos_token_id is not None:
                next_ids = next_token.squeeze(-1)
                finished = finished | (next_ids == eos_token_id)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and finished.all():
                break

        return generated[:, :seq_len]
