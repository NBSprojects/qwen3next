import torch
import torch.nn as nn 
import torch.nn.functional as F

from MoE import MoE
from gqa import GroupRopeAttention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class DecoderGQALayer(nn.Module):
    def __init__(self, num_groups, heads_per_group, active_experts, total_experts, emb_dim):
        super().__init__()

        '''
        G nombre de groupes
        H nombre de têtes par groupe
        on veut que hidden_dim des gqa vérifie : hd * G * H = emb_dim
        '''

        self.num_groups = num_groups

        assert emb_dim % num_groups == 0
        assert emb_dim % (num_groups * heads_per_group) == 0

        gqa_hd = emb_dim // (num_groups * heads_per_group)

        self.gqas = nn.ModuleList([GroupRopeAttention(emb_dim, gqa_hd, heads_per_group) for _ in range(num_groups)])
        self.moe = MoE(active_experts, total_experts, emb_dim, 4*emb_dim)

        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)

        

    def forward(self, x, kv_cache=None, use_cache=False):
        # x : [bs, seq_len, emb_dim]

        x_init = x

        x = self.norm1(x)

        new_kv_cache = None

        if use_cache:
            # kv_cache attendu : liste longueur num_groups, chaque elem None ou (K,V)
            if kv_cache is None:
                kv_cache = [None] * self.num_groups
            else:
                assert len(kv_cache) == self.num_groups

            attention_otps = []
            new_kv_cache = []
            for i, layer in enumerate(self.gqas):
                out, kv_i = layer(x, kv_cache=kv_cache[i], use_cache=True)
                attention_otps.append(out)
                new_kv_cache.append(kv_i)
        else:
            attention_otps = [layer(x) for layer in self.gqas]

        # on a une liste d'éléments de G éléments de shape [bs, seq_len, H*hd], on concatène selon la derniere dim
        x = torch.concat(attention_otps, dim=-1)

        # x : [bs, seq_len, emb_dim]

        x = x_init + x

        x_mid = x

        x = self.norm2(x)

        x, load_balancing_loss = self.moe(x)

        x_final = x_mid + x

        if use_cache:
            return x_final, load_balancing_loss, new_kv_cache
        return x_final, load_balancing_loss
