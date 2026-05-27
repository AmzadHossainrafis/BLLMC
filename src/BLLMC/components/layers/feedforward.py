import torch
import torch.nn as nn
import torch.nn.functional as F
from BLLMC.components.layers.activations import GELU


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)

    def __str__(self):
        return f"FeedForward({self.layers})"


class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.emb_dim = config.emb_dim
        self.gate = nn.Linear(config.emb_dim, config.num_experts, bias=False, dtype=config.dtype)

        self.fc1 = nn.ModuleList([nn.Linear(config.emb_dim, config.moe_hidden_dim, bias=False, dtype=config.dtype)
                                  for _ in range(config.num_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(config.emb_dim, config.moe_hidden_dim, bias=False, dtype=config.dtype)
                                  for _ in range(config.num_experts)])
        self.fc3 = nn.ModuleList([nn.Linear(config.moe_hidden_dim, config.emb_dim, bias=False, dtype=config.dtype)
                                  for _ in range(config.num_experts)])

    def forward(self, x):
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).flatten()
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = F.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch, seq_len, self.emb_dim)