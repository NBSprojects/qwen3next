import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class GroupRopeAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_heads, qk_norm=False, max_seq_len=2048  ):
        '''
        représente un groupe entier

        embedding dim la taille de l'embeddig
        hidden dime la dim de l'espace latent
        num_heads le nombre de tetes dans ce groupe (en gros = H/G)
        '''


        super().__init__()


        self.hd = hidden_dim
        self.qk_norm = qk_norm
        self.num_heads = num_heads


        self.key = nn.Linear(emb_dim, hidden_dim, bias=False)
        # on concatene tout les Wq ensemble pour rendre le calcul efficace
        self.query = nn.Linear(emb_dim, hidden_dim * num_heads, bias=False)
        self.value = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.theta_base = 10000.0 # RoPE

        self.max_seq_len = int(max_seq_len)

        dim = self.hd
        inv_freq = 1.0 / (self.theta_base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(self.max_seq_len, dtype=inv_freq.dtype)
        freqs_outer = torch.outer(t, inv_freq)
        freqs_cis = torch.repeat_interleave(freqs_outer, 2, dim=-1)

        self.register_buffer("rope_cos", torch.cos(freqs_cis), persistent=False)
        self.register_buffer("rope_sin", torch.sin(freqs_cis), persistent=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.bool), diagonal=1),
            persistent=False
        )




    def compute_rope_mats(self, x, start_pos: int = 0):
        '''
        calcule 2 torch matrices de taille (seq_len, self.hd)
        la premiere vaut Rc(i, j) = cos(i * theta_((j // 2) + 1))
        la deuxieme vaut Rs(i, j) = sin(i * theta_((j // 2) + 1))
        où theta_k = self.theta_val ** (2*k / self.hd)

        retourne Rc, Rs de shape (1, 1, L, D) pour appliquer RoPE
        sur un tenseur de shape (..., L, D). start_pos permet l'offset (KV cache).
        '''
        # IMPORTANT: L est l'avant-dernière dimension (Seq)
        seq_len = x.shape[-2]
        device = x.device
        dtype = x.dtype

        dim = self.hd
        freqs = 1.0 / (self.theta_base ** (torch.arange(0, dim, 2, device=device).float() / dim))

        # positions absolues : [start_pos, ..., start_pos + seq_len - 1]
        t = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=freqs.dtype)
        freqs_outer = torch.outer(t, freqs)  # (seq_len, dim/2)

        freqs_cis = torch.repeat_interleave(freqs_outer, 2, dim=-1)  # (seq_len, dim)

        Rc = torch.cos(freqs_cis).unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        Rs = torch.sin(freqs_cis).unsqueeze(0).unsqueeze(0)

        return Rc.to(dtype), Rs.to(dtype)

        
    def apply_rope(self, x, Rc, Rs):
        # multiplier x par Rc element wise
        # y = permuter chaque ligne de x chaque paire d'élément de cette ligne puis faire * (-1) sur les positions impaires
        # multiplier y par Rs element wise
        # retourner x + y 

        # le slicing ne pose pas de problèmes pour la compilation torch.compile

        # Rotation (-x_odd, x_even)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        
        # Pour reconstruire, on interleave
        # Attention : torch.stack puis flatten est plus sûr pour l'entrelacement
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = -x_odd
        x_rot[..., 1::2] = x_even
        
        # Note: Le broadcasting de Rc/Rs (1,1,L,D) s'appliquera automatiquement
        return x * Rc + x_rot * Rs



    def forward(self, x, kv_cache=None, use_cache: bool = False):
        """
        x: [B, L, emb_dim]
        kv_cache: None ou (K_cache, V_cache)
          - K_cache: [B, 1, L_past, hd] (déjà RoPE-appliqué)
          - V_cache: [B, 1, L_past, hd]
        use_cache:
          - si True, retourne (out, (K_total, V_total))
          - sinon, retourne out
        """
        B, Lq, _ = x.shape  # Lq = longueur query (souvent 1 en génération)

        # Q : [B, Lq, num_heads*hd] -> [B, num_heads, Lq, hd]
        Qs = self.query(x).view(B, Lq, self.num_heads, self.hd).transpose(1, 2)

        # K_new, V_new : [B, 1, Lq, hd]
        K_new = self.key(x).unsqueeze(1)
        V_new = self.value(x).unsqueeze(1)

        if self.qk_norm:
            Qs = F.normalize(Qs, p=2, dim=-1)
            K_new = F.normalize(K_new, p=2, dim=-1)

        past_len = 0
        if kv_cache is not None:
            K_past, V_past = kv_cache
            past_len = K_past.shape[2]

        # on ne recalcule les matrices RoPE que si l'on a depassé le contexte max stocké dans le buffer
        if past_len + Lq <= self.max_seq_len:
            Rc = self.rope_cos[past_len: past_len + Lq].unsqueeze(0).unsqueeze(0).to(dtype=Qs.dtype)
            Rs = self.rope_sin[past_len: past_len + Lq].unsqueeze(0).unsqueeze(0).to(dtype=Qs.dtype)
        else:
            Rc, Rs = self.compute_rope_mats(Qs, start_pos=past_len)
        Qs = self.apply_rope(Qs, Rc, Rs)
        K_new = self.apply_rope(K_new, Rc, Rs)

        # Concat cache + nouveaux KV (si cache fourni)
        if kv_cache is not None:
            K = torch.cat([K_past, K_new], dim=2)   # [B, 1, L_total, hd]
            V = torch.cat([V_past, V_new], dim=2)   # [B, 1, L_total, hd]
        else:
            K, V = K_new, V_new

        Lk = K.shape[2]  # longueur key totale

        # scores: [B, num_heads, Lq, Lk]
        if(self.qk_norm):
            scores = torch.einsum('bgld,bgjd->bglj', Qs, K)
        else:
            scores = torch.einsum('bgld,bgjd->bglj', Qs, K) / math.sqrt(self.hd)

        if past_len == 0 and Lq == Lk and Lk <= self.max_seq_len:
            scores = scores.masked_fill(self.causal_mask[:Lq, :Lk], float('-inf'))
        else:
            causal_mask = torch.triu(
                torch.ones((Lq, Lk), device=x.device, dtype=torch.bool),
                diagonal=1 + past_len
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # softmax en float32 pour éviter les problèmes de stabilité
        atts = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)  # [B, num_heads, Lq, Lk]

        # out: [B, num_heads, Lq, hd]
        otps = torch.einsum('bgij,bgjk->bgik', atts, V)

        # [B, Lq, num_heads*hd]
        otps = otps.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.hd)

        if use_cache:
            return otps, (K, V)
        return otps

