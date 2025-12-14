# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional 

from decoder import DecoderGQALayer, RMSNorm


class DecoderOnlyLM(nn.Module):
    """
    Modèle decoder-only pour du next-token prediction, basé sur :
      - DecoderGQALayer (GQA + MoE)
      - RoPE dans les têtes d'attention
      - FFN MoE type Switch / DeepSeek

    Paramètres principaux :
      vocab_size        : taille du vocabulaire
      emb_dim           : dimension des embeddings / état du modèle
      n_layers          : nombre de couches DecoderGQALayer
      num_groups        : nombre de groupes G (voir DecoderGQALayer)
      heads_per_group   : nombre de têtes par groupe H
      active_experts    : k, nombre d'experts actifs par token
      total_experts     : nombre total d'experts MoE
      tie_embeddings    : si True, lm_head partage les poids avec l'embedding
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_layers: int,
        num_groups: int,
        heads_per_group: int,
        active_experts: int,
        total_experts: int,
        tie_embeddings: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.num_groups = num_groups

        # Embedding de tokens
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)

        # Stack de couches Decoder (Attention GQA + MoE)
        self.layers = nn.ModuleList(
            [
                DecoderGQALayer(
                    num_groups=num_groups,
                    heads_per_group=heads_per_group,
                    active_experts=active_experts,
                    total_experts=total_experts,
                    emb_dim=emb_dim,
                )
                for _ in range(n_layers)
            ]
        )

        # Norm finale de la sortie
        self.norm_out = RMSNorm(emb_dim)

        # Tête de langage (projection vers vocabulaire)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Option : tie embeddings (comme GPT / LLaMA)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    # ------------------------------------------------------------------ #
    #  FORWARD : passe standard, avec ou sans KV cache
    # ------------------------------------------------------------------ #

    def forward(self, input_ids, kv_cache=None, use_cache: bool = False):
        """
        input_ids : LongTensor [B, L]
        kv_cache  :
          - None
          - ou liste de longueur n_layers,
            chaque entrée étant:
              * None, ou
              * liste de longueur num_groups pour cette couche,
                où chaque élément est:
                    - None, ou
                    - tuple (K_cache, V_cache) de GroupRopeAttention
                      (shapes [B, 1, L_past, hd])

        retourne :
          - logits : [B, L, vocab_size]
          - load_balancing_loss : scalaire (moyenne des MoE sur les couches)
          - (optionnel) new_kv_cache si use_cache=True
        """
        if input_ids.dim() == 1:
            # On accepte [L] -> [1, L]
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape

        # Embedding [B, L, D]
        x = self.tok_emb(input_ids)

        load_balancing_losses = []

        if use_cache:
            # Structure : liste[n_layers] de caches de couche
            if kv_cache is None:
                kv_cache = [None] * self.n_layers
            else:
                assert len(kv_cache) == self.n_layers, "kv_cache mal dimensionné"
            new_kv_cache = []

        # Passage dans chaque couche DecoderGQALayer
        for layer_idx, layer in enumerate(self.layers):
            if use_cache:
                x, lb_loss, layer_cache = layer(
                    x, kv_cache=kv_cache[layer_idx], use_cache=True
                )
                new_kv_cache.append(layer_cache)
            else:
                x, lb_loss = layer(x, kv_cache=None, use_cache=False)

            load_balancing_losses.append(lb_loss)

        # Norm finale + tête de langage
        x = self.norm_out(x)           # [B, L, D]
        logits = self.lm_head(x)       # [B, L, vocab_size]

        if load_balancing_losses:
            load_balancing_loss = torch.stack(load_balancing_losses).mean()
        else:
            load_balancing_loss = torch.tensor(0.0, device=logits.device)

        if use_cache:
            return logits, load_balancing_loss, new_kv_cache

        return logits, load_balancing_loss

    # ------------------------------------------------------------------ #
    #  SAMPLING : top-k / top-p, avec ou sans KV cache
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        logits : [B, vocab_size]

        - temperature = 0 ou None -> greedy (argmax)
        - top_k : si non None, garde les top_k logits
        - top_p : nucleus sampling (garde un préfixe cumulant ~top_p de probas)

        retourne : next_token_ids [B]
        """
        # Greedy si temp <= 0
        if temperature is None or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        # Température
        logits = logits / temperature

        vocab_size = logits.size(-1)

        # Top-k
        if top_k is not None and 0 < top_k < vocab_size:
            values, indices = torch.topk(logits, top_k, dim=-1)
            mask = logits.new_full(logits.shape, float("-inf"))
            mask.scatter_(1, indices, values)
            logits = mask

        # Top-p (nucleus)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, dim=-1, descending=True
            )  # [B, V]
            probs = F.softmax(sorted_logits, dim=-1)
            cdf = probs.cumsum(dim=-1)

            # On masque dès que la CDF dépasse top_p
            mask = cdf > top_p
            # On garde au moins un token par ligne
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

            # On remet dans l'ordre original
            logits = logits.new_full(logits.shape, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)

        # Sécurité numérique : si jamais NaN, on remplace par uniforme
        if torch.isnan(probs).any():
            probs = torch.where(
                torch.isnan(probs),
                torch.full_like(probs, 1.0 / probs.size(-1)),
                probs,
            )

        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_kv_cache: bool = True,
        eos_token_id: Optional[int] = None,
        return_cache: bool = False,
    ):

        """
        Génération autoregressive (next-token prediction) à partir d'un prompt.

        input_ids      : [B, L] ou [L]
        max_new_tokens : nombre maximum de nouveaux tokens à générer
        temperature    : float (0 => greedy)
        top_k          : int ou None
        top_p          : float ou None (0<top_p<=1)
        use_kv_cache   : si True, utilise le cache KV pour les passes suivantes
        eos_token_id   : si donné, arrêt anticipé quand tous les batchs génèrent EOS
        return_cache   : si True, retourne aussi le kv_cache final

        retourne :
          - generated : [B, L + T] (T <= max_new_tokens)
          - éventuellement kv_cache si return_cache=True
        """
        self.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, L]

        generated = input_ids
        kv_cache = None

        for _ in range(max_new_tokens):
            if use_kv_cache:
                if kv_cache is None:
                    # Premier passage : on traite tout le prompt
                    logits, _, kv_cache = self.forward(
                        generated, kv_cache=None, use_cache=True
                    )
                else:
                    # Ensuite : on ne passe que le dernier token
                    logits, _, kv_cache = self.forward(
                        generated[:, -1:], kv_cache=kv_cache, use_cache=True
                    )
            else:
                # Pas de cache : on repasse tout le contexte à chaque étape
                logits, _ = self.forward(generated, kv_cache=None, use_cache=False)

            # On ne garde que les logits du dernier token
            logits_last = logits[:, -1, :]  # [B, vocab_size]

            # Sampling top-k / top-p / temperature
            next_token = self._sample_from_logits(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # [B]

            generated = torch.cat(
                [generated, next_token.unsqueeze(-1)], dim=-1
            )  # [B, L+1]

            # Arrêt anticipé si eos_token_id fourni
            if eos_token_id is not None:
                if (generated[:, -1] == eos_token_id).all():
                    break

        if return_cache:
            return generated, kv_cache
        return generated
