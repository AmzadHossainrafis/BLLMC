import torch
import torch.nn as nn
import torch.nn.functional as F
from BLLMC.components.layers.activations import GELU, swiglu


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
        self.gate = nn.Linear(
            config.emb_dim, config.num_experts, bias=False, dtype=config.dtype
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    config.emb_dim,
                    config.moe_hidden_dim,
                    bias=False,
                    dtype=config.dtype,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(
                    config.emb_dim,
                    config.moe_hidden_dim,
                    bias=False,
                    dtype=config.dtype,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.fc3 = nn.ModuleList(
            [
                nn.Linear(
                    config.moe_hidden_dim,
                    config.emb_dim,
                    bias=False,
                    dtype=config.dtype,
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x):
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(
            batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype
        )

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
            gate = self.fc1[expert_id](expert_input).clamp(max=7.0)
            up = self.fc2[expert_id](expert_input).clamp(min=-7.0, max=7.0)
            hidden = F.silu(gate) * up
            expert_out = self.fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx),
                dim=-1,
                index=slot_indices,
            ).squeeze(-1)

            out_flat.index_add_(
                0, selected_idx, expert_out * selected_probs.unsqueeze(-1)
            )

        return out_flat.reshape(batch, seq_len, self.emb_dim)

    def __str__(self):
        return f"MoEFeedForward(emb_dim={self.emb_dim}, num_experts={self.num_experts}, num_experts_per_tok={self.num_experts_per_tok})"

    def __repr__(self):
        return self.__str__()


class Llama2FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.emb_dim, config.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.emb_dim, bias=False)

    def forward(self, x):
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)

    def __str__(self):
        return f"Llama2FeedForward(emb_dim={self.config.emb_dim}, ffn_hidden_dim={self.config.ffn_hidden_dim})"

    def __repr__(self):
        return self.__str__()


class SingleExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.emb_dim, config.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.emb_dim, bias=False)

    def forward(self, x):
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)

    def __str__(self):
        return f"SingleExpert(emb_dim={self.config.emb_dim}, ffn_hidden_dim={self.config.ffn_hidden_dim})"

    def __repr__(self):
        return self.__str__()


class GPTOssFeedForward(nn.Module):
    """
    GPT-OSS FeedForward Network implementing MoE with custom SwiGLU.

    Supports interleaved splitting, clamping limits, scaling and bias adjustments.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.swiglu_limit = getattr(config, "swiglu_limit", 7.0)
        self.emb_dim = config.emb_dim

        # Check if distributed is initialized
        import torch.distributed as dist

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.gate = nn.Linear(
            config.emb_dim, config.num_experts, bias=True, dtype=config.dtype
        )

        hidden_dim = getattr(config, "moe_hidden_dim", config.emb_dim)
        # Store experts as sequential containers of two linear layers
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        config.emb_dim,
                        (hidden_dim * 2) // self.world_size,
                        bias=True,
                        dtype=config.dtype,
                    ),
                    nn.Linear(
                        hidden_dim // self.world_size,
                        config.emb_dim,
                        bias=True,
                        dtype=config.dtype,
                    ),
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)

        # Compute gating scores
        scores = self.gate(x_flat)

        # Get top-k experts
        topk_scores, topk_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1, sorted=True
        )
        topk_probs = torch.softmax(topk_scores, dim=-1)

        out_flat = torch.zeros_like(x_flat)

        # Flatten routing metadata
        topk_indices_flat = topk_indices.view(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.view(-1, self.num_experts_per_tok)

        # Process each expert
        for expert_idx in range(self.num_experts):
            mask = (topk_indices_flat == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            token_indices = torch.where(mask)[0]
            expert_pos = (topk_indices_flat[token_indices] == expert_idx).nonzero(
                as_tuple=True
            )[1]

            expert_input = x_flat[token_indices]
            weights = topk_probs_flat[token_indices, expert_pos]

            # Forward through the sequential expert layers:
            # 1. Linear projection (hidden_size -> moe_hidden_dim * 2)
            expert_out = self.experts[expert_idx][0](expert_input)
            # 2. Custom SwiGLU activation (with limit clamping and +1 bias)
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            # 3. Down projection (moe_hidden_dim -> hidden_size)
            expert_out = self.experts[expert_idx][1](expert_out)

            out_flat[token_indices] += expert_out * weights.unsqueeze(-1)

        # Distributed all-reduce sum across world size ranks
        if self.world_size > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(out_flat, op=dist.ReduceOp.SUM)

        return out_flat.reshape(batch, seq_len, self.emb_dim)

    def __str__(self):
        return (
            f"GPTOssFeedForward("
            f"emb_dim={self.emb_dim}, "
            f"num_experts={self.num_experts}, "
            f"num_experts_per_tok={self.num_experts_per_tok}, "
            f"swiglu_limit={self.swiglu_limit})"
        )

    def __repr__(self):
        return self.__str__()
