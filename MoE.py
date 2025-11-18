import torch
import torch.nn as nn
import torch.nn.functional as F

'''
references :
https://arxiv.org/pdf/2401.06066

'''

class FFN(nn.Module):
    def __init__(self, in_d, hd):
        super().__init__()
        # we use SwiGLU FFN
        self.fc1 = nn.Linear(in_d, hd, bias=False)
        self.gating = nn.Linear(in_d, hd, bias=False)
        self.fc2 = nn.Linear(hd, in_d, bias=False)


    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.gating(x)
        x3 = F.silu(x1) * x2 # entry wise
        x4 = self.fc2(x3)

        return x4


class MoE(nn.Module):
    def __init__(self, k, num_exp, in_d, hd):
        super().__init__()
        '''
        k : number of selected experts 
        num_exp : total number of experts
        in_d : embedding dimension
        hd : hidden dim of MLP layer
        '''
        self.k = k
        self.num_exp = num_exp
        self.experts = nn.ModuleList([FFN(in_d, hd) for _ in range(num_exp)])
        self.gate_layer = nn.Linear(in_d, num_exp, bias=False)

    def forward(self, x):
        # x: [bs, seq_len, in_d]
        bs, seq_len, in_d = x.shape
        N = bs * seq_len

        logits = self.gate_layer(x)                     # [bs, seq_len, num_exp]
        probs = F.softmax(logits, dim=-1)               # for routing

        # probs_lbl for balancing loss (detached input)
        probs_lbl = F.softmax(self.gate_layer(x.detach()), dim=-1)

        topk_vals, topk_inds = torch.topk(probs, self.k, dim=-1)  # [bs, seq_len, k]
        cl_values = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # [bs, seq_len, k]

        # OPTIMIZED INFERENCE THROUGH FFNs
        # the idea is to group tokens by attributed experts
        # in a multi gpu context we load experts on different gpus to speed up the process

        x_flat = x.view(N, in_d)                        # [N, in_d]
        topk_inds_flat = topk_inds.view(N * self.k)     # [N*k]
        cl_values_flat = cl_values.view(N * self.k)     # [N*k]
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.k, -1).reshape(N * self.k, in_d)  # [N*k, in_d]

        # Sort by expert index to group tokens
        sorted_inds, sort_indices = torch.sort(topk_inds_flat)
        sorted_x = x_expanded[sort_indices]
        sorted_weights = cl_values_flat[sort_indices]

        output_flat = torch.zeros(N * self.k, in_d, device=x.device)
        unique_inds, counts = torch.unique_consecutive(sorted_inds, return_counts=True)

        start = 0
        for exp_id, count in zip(unique_inds, counts):
            if count > 0:
                tokens = sorted_x[start:start + count]
                expert_out = self.experts[exp_id](tokens)  # [count, in_d]
                # Re-weight and place back in original dispatch order
                weighted_out = expert_out * sorted_weights[start:start + count].unsqueeze(1)
                output_flat[sort_indices[start:start + count]] = weighted_out
            start += count

        # --- Recombine: sum over k experts per token ---
        output_flat = output_flat.view(N, self.k, in_d)
        moe_output = output_flat.sum(dim=1).view(bs, seq_len, in_d)  # [bs, seq_len, in_d]

        # Load balancing loss terms
        p = probs_lbl.mean(dim=(0, 1))  # [num_exp]

        # f is not differentiable so we consider it constant
        with torch.no_grad():
            indices_flat = topk_inds.view(-1)  # [N * k]
            total_slots = indices_flat.numel()
            f_counts = torch.zeros(self.num_exp, dtype=torch.float32, device=x.device)
            f_counts.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float32))
            f = f_counts / total_slots  # [num_exp]

        balancing_loss = torch.sum(p * f)

        return moe_output, balancing_loss

