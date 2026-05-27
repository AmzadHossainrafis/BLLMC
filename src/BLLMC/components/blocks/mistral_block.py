import torch.nn as nn


from BLLMC.components.attention.sw_attention import SlidingWindowAttention
from BLLMC.components.layers.feedforward import MoEFeedForward


class MistralBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms_norm1 = nn.RMSNorm(config.emb_dim)
        self.mha = SlidingWindowAttention(config)
        self.dropout = nn.Dropout(config.drop_rate)
        self.rms_norm2 = nn.RMSNorm(config.emb_dim)
        self.ffn = MoEFeedForward(config)

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.rms_norm1(x)
        x = self.mha(x, use_cache=use_cache)
        x = self.dropout(x)
        x = shortcut + x
        shortcut = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = shortcut + x
        return x

    def reset_cache(self):
        if hasattr(self.mha, "reset_cache"):
            self.mha.reset_cache()

    def __str__(self):
        return "MistralBlock(mha={}, ffn={}, rms_norm1={}, rms_norm2={}, dropout={})".format(
            self.mha, self.ffn, self.rms_norm1, self.rms_norm2, self.dropout
        )

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from BLLMC.components.config import mistral_config

    config = mistral_config()
    mistral_block = MistralBlock(config)
    print(mistral_block)
