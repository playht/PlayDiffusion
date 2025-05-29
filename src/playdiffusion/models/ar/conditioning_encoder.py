# base model for ar
import torch
from torch import nn

from playdiffusion.utils.model_util import QKVAttentionLegacy, zero_module, normalization

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        mlp_mult=4,
        do_checkpoint=True,
        zero_init_residual=True,
        mel_sample_rate=24000,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.attn_norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.attn_proj_out = (
            zero_module(nn.Conv1d(channels, channels, 1)) if zero_init_residual else nn.Conv1d(channels, channels, 1)
        )

        self.ffnorm = normalization(channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels),  # This is the positional embedding.
            nn.Conv1d(channels, channels * mlp_mult, 1),
            nn.GELU(),
            zero_module(nn.Conv1d(channels * mlp_mult, channels, 1)),
        )

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.attn_norm(x)
        with torch.autocast(h.device.type):
            qkv = self.qkv(h)
            h = self.attention(qkv, mask)
            h = self.attn_proj_out(h)
        x = x + h#.float()
        h = self.ffnorm(x)
        with torch.autocast(h.device.type):
            h = self.mlp(h)
        x = x + h#.float()
        return x.reshape(b, c, *spatial)


class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, do_checkpointing=False, mel_sample_rate=24000):
        super().__init__()
        attn = []
        self.init = nn.Sequential(
            nn.Conv1d(spec_dim, embedding_dim, kernel_size=5, padding=3, stride=2),
        )
        for a in range(attn_blocks):
            attn.append(
                AttentionBlock(
                    embedding_dim,
                    embedding_dim // 64,
                    mlp_mult=4,
                    do_checkpoint=do_checkpointing,
                    zero_init_residual=True,
                )
            )
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mel_sample_rate = mel_sample_rate

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
