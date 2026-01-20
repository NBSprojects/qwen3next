import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
references :
https://arxiv.org/pdf/2101.03961 (switch transformers)
https://arxiv.org/pdf/2401.06066 (DeepSeek MoE)

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
        """
        k : number of selected experts
        num_exp : total number of experts
        in_d : embedding dimension
        hd : hidden dim of MLP layer
        """
        self.k = k
        self.num_exp = num_exp
        self.in_d = in_d

        self.experts = nn.ModuleList([FFN(in_d, hd) for _ in range(num_exp)])
        self.gate_layer = nn.Linear(in_d, num_exp, bias=False)

        # --- Capacity routing knobs (keep signature; tune by setting attributes after init) ---
        self.capacity_factor = 1.25   # typical values: 1.0, 1.25, 2.0 ...
        self.min_capacity = 4         # avoid tiny capacities on very small batches
        self.capacity = None          # if set (int): overrides computed capacity

        self.register_buffer(
            "last_tokens_per_expert",
            torch.zeros(num_exp, dtype=torch.float32),
            persistent=False
        )

    def _compute_capacity(self, N_tokens: int) -> int:
        """
        Compute per-expert capacity in number of *slots*.
        For top-k routing, total slots = N_tokens * k.
        """
        if self.capacity is not None:
            cap = int(self.capacity)
            return max(cap, 1)

        # average slots per expert (ceil)
        total_slots = N_tokens * self.k
        avg = (total_slots + self.num_exp - 1) // self.num_exp
        cap = int(math.ceil(avg * float(self.capacity_factor)))
        cap = max(cap, int(self.min_capacity))
        return cap

    def forward(self, x):
        # x: [bs, seq_len, in_d]
        bs, seq_len, in_d = x.shape
        assert in_d == self.in_d, f"Expected in_d={self.in_d}, got {in_d}"
        N = bs * seq_len
        NK = N * self.k

        # --- Router (dtype-safe, memory-friendly) ---
        w = self.gate_layer.weight  # peut être fp32 si tu as appliqué le changement 7

        w_for_mm = w.to(dtype=x.dtype) if w.dtype != x.dtype else w

        logits = F.linear(x, w_for_mm)  # logits en bf16
        probs  = F.softmax(logits.float(), dim=-1)  # softmax en fp32 (stable)

        logits_lbl = F.linear(x.detach(), w_for_mm)
        probs_lbl  = F.softmax(logits_lbl.float(), dim=-1)

        topk_vals, topk_inds = torch.topk(probs, self.k, dim=-1)  # [bs, seq_len, k]
        cl_values = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)  # [bs, seq_len, k]

        # --- Flatten into (token, expert) dispatch entries ---
        x_flat = x.reshape(N, in_d)  # [N, in_d]
        x_expanded = (
            x_flat.unsqueeze(1)
                 .expand(-1, self.k, -1)
                 .reshape(NK, in_d)
        )  # [N*k, in_d]

        exp_ids = topk_inds.reshape(NK).to(torch.int64)      # [N*k]
        weights = cl_values.reshape(NK)                      # [N*k]

        # --- Capacity ---
        capacity = self._compute_capacity(N)
        dummy = self.num_exp * capacity  # reserve last index as "overflow sink"

        # --- Sort by (expert_id, dispatch_index) to compute position_in_expert deterministically ---
        dispatch_idx = torch.arange(NK, device=x.device, dtype=torch.int64)
        key = exp_ids * NK + dispatch_idx  # lexicographic sort: expert_id primary, dispatch_idx secondary
        sort_perm = torch.argsort(key)     # [N*k]

        sorted_exp = exp_ids.index_select(0, sort_perm)     # [N*k]
        sorted_x = x_expanded.index_select(0, sort_perm)    # [N*k, in_d]
        sorted_w = weights.index_select(0, sort_perm)       # [N*k]

        # --- counts per expert (fixed shape [num_exp]) ---
        counts = torch.zeros(self.num_exp, device=x.device, dtype=torch.int64)
        ones = torch.ones_like(sorted_exp, dtype=torch.int64)
        counts.scatter_add_(0, sorted_exp, ones)

        # starts[e] = starting offset of expert e block in the sorted list
        starts = torch.cumsum(counts, dim=0) - counts  # [num_exp]

        # local_pos[i] = index of this dispatch entry within its expert block
        sorted_pos = torch.arange(NK, device=x.device, dtype=torch.int64)  # [N*k]
        local_pos = sorted_pos - starts.index_select(0, sorted_exp)        # [N*k]

        # keep only the first `capacity` entries per expert
        keep = local_pos < capacity  # [N*k] boolean

        # slot = expert_id * capacity + local_pos, but map overflow to `dummy`
        slot = sorted_exp * capacity + local_pos  # may exceed range for overflow
        slot = torch.where(keep, slot, torch.full_like(slot, dummy))  # [N*k] in [0, dummy]

        # --- Dispatch buffer (flat) with extra dummy row ---
        # buf_x_flat has shape [num_exp*capacity + 1, in_d]
        buf_x_flat = x.new_zeros((dummy + 1, in_d))
        # Index-copy is row-wise; duplicates only happen at dummy (overflow), which we ignore later.
        buf_x_flat.index_copy_(0, slot, sorted_x)

        # Remove dummy row and reshape to [num_exp, capacity, in_d]
        buf_x = buf_x_flat[:-1].reshape(self.num_exp, capacity, in_d)

        # --- Expert compute (static loop, good for torch.compile) ---
        buf_out = torch.empty_like(buf_x)
        for e in range(self.num_exp):
            buf_out[e] = self.experts[e](buf_x[e])  # [capacity, in_d]

        buf_out_flat = buf_out.reshape(self.num_exp * capacity, in_d)

        # Add dummy output row of zeros so we can safely gather overflow slots
        buf_out_full = x.new_zeros((dummy + 1, in_d))
        buf_out_full[:-1].copy_(buf_out_flat)

        # --- Gather back, apply weights (mask overflow by zeroing weights) ---
        w_masked = (sorted_w * keep.to(sorted_w.dtype)).to(buf_out_full.dtype)  # -> bf16
        out_sorted = buf_out_full.index_select(0, slot)                         # bf16
        out_sorted = out_sorted * w_masked.unsqueeze(1)                         # reste bf16


        # --- Unsort to original dispatch order ---
        out_dispatch = out_sorted.new_empty((NK, in_d))
        out_dispatch.index_copy_(0, sort_perm, out_sorted)  # out_dispatch[sort_perm[i]] = out_sorted[i]

        # --- Recombine: sum over k experts per token ---
        out_dispatch = out_dispatch.reshape(N, self.k, in_d)
        moe_output = out_dispatch.sum(dim=1).reshape(bs, seq_len, in_d)

        p = probs_lbl.mean(dim=(0, 1))  # [num_exp]

        with torch.no_grad():
            # f : fraction de slots alloués par expert (pour le balancing loss)
            indices_flat = topk_inds.reshape(-1)  # [N*k]
            total_slots = indices_flat.numel()
            f_counts = torch.zeros(self.num_exp, dtype=torch.float32, device=x.device)
            f_counts.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float32))
            f = f_counts / float(total_slots)

            # Nouveau : nombre de "tokens effectifs" (slots non-overflow) par expert
            # sorted_exp : expert_id pour chaque dispatch entry (après tri)
            # keep       : booléen -> True si l'entrée n'est pas overflow (local_pos < capacity)
            effective_counts = torch.zeros(self.num_exp, dtype=torch.float32, device=x.device)
            effective_counts.scatter_add_(0, sorted_exp, keep.to(torch.float32))
            # On mémorise pour que le code de train puisse les lire
            self.last_tokens_per_expert.copy_(effective_counts)

        balancing_loss = self.num_exp * torch.sum(p * f)

        return moe_output.to(x.dtype), balancing_loss

