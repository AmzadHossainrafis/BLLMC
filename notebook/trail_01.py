###############
# Chapter 3
#####################################


import torch
import torch.nn as nn


class MultiHeadAttentionWithSWA(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False, sliding_window_size=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=True):
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        if use_cache:
            old_cache_k, old_cache_v = self.cache_k, self.cache_v
            old_len = 0 if old_cache_k is None else old_cache_k.size(1)
            print(f"old len: {old_len}")
            print(f"Keys new: {keys_new.size(1)}")
            if old_cache_k is None:
                combined_k, combined_v = keys_new, values_new
            else:
                print("Old cache k:", old_cache_k.size())
                combined_k = torch.cat([old_cache_k, keys_new], dim=1)
                print(f"Combined k: {combined_k.size()}")
                combined_v = torch.cat([old_cache_v, values_new], dim=1)

            keys, values = combined_k, combined_v
            if self.sliding_window_size is not None:
                attn_keep = min(keys.size(1), self.sliding_window_size + num_tokens - 1)
                keys = keys[:, -attn_keep:, :, :]
                values = values[:, -attn_keep:, :, :]
                cache_keep = min(combined_k.size(1), self.sliding_window_size)
                self.cache_k = combined_k[:, -cache_keep:, :, :]
                self.cache_v = combined_v[:, -cache_keep:, :, :]
            else:
                self.cache_k, self.cache_v = combined_k, combined_v

            dropped = combined_k.size(1) - keys.size(1)
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
        else:
            keys, values = keys_new, values_new

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        # Determine absolute positions for q and k
        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0
        q_positions = torch.arange(
            q_start, q_start + num_tokens_Q, device=device, dtype=torch.long
        )
        k_positions = torch.arange(
            k_start, k_start + num_tokens_K, device=device, dtype=torch.long
        )
        # Sliding window width
        W = (
            num_tokens_K + 1
            if self.sliding_window_size is None
            else int(self.sliding_window_size)
        )
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)
        mask_bool = (diff < 0) | (diff >= W)
        if use_cache:
            self.ptr_current_pos += num_tokens_Q
        else:
            self.ptr_current_pos = 0

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


###
if __name__ == "__main__":
    d_in = 128
    d_out = 128
    num_heads = 4
    sliding_window_size = 10
    mha = MultiHeadAttentionWithSWA(
        d_in,
        d_out,
        dropout=0.0,
        num_heads=num_heads,
        sliding_window_size=sliding_window_size,
    )

    # 1. First forward pass (generates cache)
    for i in range(3):
        print(f"Block {i+1}")

        seq_len = 128
        x1 = torch.randn(2, seq_len, d_in)
        out1 = mha(x1, use_cache=True)
        print("Block 1 output:", out1.shape)

        # 2. Next forward pass using KV cache (longer effective context)
        # Now model can use cache from first pass
        x2 = torch.randn(2, 128, d_in)
        out2 = mha(x2, use_cache=True)  # pass True to enable KV cache
        print("Block 2 output:", out2.shape)

        # 3. Run without caching - window will reset
        x3 = torch.randn(2, 128, d_in)
        out3 = mha(x3, use_cache=True)  # window resets, no cache used
        print("Block 3 output (no cache):", out3.shape)
