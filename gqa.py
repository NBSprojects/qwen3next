import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class GroupRopeAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_heads, qk_norm=False):
        '''
        '''
        # embedding dim la taille de l'embeddig
        # hidden dime la dim de l'espace latent
        # num_heads le nombre de tetes dans ce groupe (en gros = H/G)

        super().__init__()


        self.hd = hidden_dim
        self.qk_norm = qk_norm
        self.num_heads = num_heads


        self.key = nn.Linear(emb_dim, hidden_dim, bias=False)
        # on concatene tout les Wq ensemble pour rendre le calcul efficace
        self.query = nn.Linear(emb_dim, hidden_dim * num_heads, bias=False)
        self.value = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.theta_base = 10000.0 


    def compute_rope_mats(self, x):
        '''
        calcule 2 torch matrices de taille (seq_len, self.hd)
        la premiere vaut Rc(i, j) = cos(i * theta_((j // 2) + 1))
        la deuxieme vaut Rs(i, j) = sin(i * theta_((j // 2) + 1))
        où theta_k = self.theta_val ** (2*k / self.hd)
        '''
        seq_len = x.shape[1]
        device = x.device
        dtype = x.dtype 

        # Fréquences : dim/2 valeurs
        dim = self.hd
        freqs = 1.0 / (self.theta_base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        
        t = torch.arange(seq_len, device=device, dtype=freqs.dtype)
        freqs_outer = torch.outer(t, freqs) # (seq_len, dim/2)
        
        # CORRECTION MATHS : On répète pour que (0,1), (2,3) partagent la même freq
        freqs_cis = torch.repeat_interleave(freqs_outer, 2, dim=-1) # (seq_len, dim)
        
        # Broadcast pour correspondre aux dimensions (Batch, Head, Seq, Dim)
        # Cos et Sin
        Rc = torch.cos(freqs_cis).unsqueeze(0).unsqueeze(0) # (1, 1, L, D)
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



    def forward(self, x):

        # x : [bs, seq_len, emb_dim]

        B, L, _ = x.shape
        
        # Q : [B, L, G*H] -> [B, L, G, H] -> [B, G, L, H]
        Qs = self.query(x).view(B, L, self.num_heads, self.hd).transpose(1, 2)
        # K, V : [B, L, H] -> [B, 1, L, H] (Broadcast pour le groupe)
        K = self.key(x).unsqueeze(1) 
        V = self.value(x).unsqueeze(1)

        if self.qk_norm:
            Qs = F.normalize(Qs, p=2, dim=-1)
            K  = F.normalize(K,  p=2, dim=-1)

        Rc, Rs = self.compute_rope_mats(Qs)
        Qs = self.apply_rope(Qs, Rc, Rs)
        K = self.apply_rope(K, Rc, Rs)

        # Qs: [B, G, L, D], K: [B, 1, L, D]
        scores = torch.einsum('bgld,bgjd->bglj', Qs, K) / math.sqrt(self.hd)

        causal_mask = torch.triu(torch.ones((L, L), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))

        atts = torch.softmax(scores, dim=-1) # [B, G, L, L]

        otps = torch.einsum('bgij,bgjk->bgik', atts, V) # [B, G, L, D]

        otps = otps.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.hd)
        
        return otps

