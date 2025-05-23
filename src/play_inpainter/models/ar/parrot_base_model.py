# base model for ar
import torch
from torch import nn

from play_inpainter.utils.model_util import QKVAttentionLegacy, zero_module, normalization

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        mlp_mult=4,
        do_checkpoint=True,
        zero_init_residual=True,
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
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, do_checkpointing=False):
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

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2)

class BaseModel(nn.Module):
    def __init__(
        self,
        mel_sample_rate=22050,
        layers=40,
        model_dim=1024,
        mlp_mult=6,
        mel_dim=80,
        voice_encoder_depth=6,
        max_text_tokens=650,
        extra_mel_tokens=300,
        max_mel_tokens=1200,
        mel_extra_offset=3,
        number_text_tokens=256,
        number_mel_tokens=8193,
        version='v2',
        # THESE SHOULD BE IN SAMPLING ARGS, but due to the limitations of FT, we have them here
        # The candidate selection logic in our sampling heuristic relies on a way to
        # detect skipped tokens. This is done by looking at one of our attention maps,
        # which will show an approximate correlation between text tokens and MEL tokens
        # as a 2D chart per candidate sequence. A skipped token will be seen as a
        # discontinuity in this chart. Since we have n_layers * n_heads attention maps,
        # we need to choose the one that best represents this correlation. This was done
        # experimentally and their indices are configured below.
        sampling_attn_layer_idx: int=3,
        sampling_attn_head_idx: int=12,
        sampling_attn_threshold: float=0.4
    ):
        super().__init__()
        self.mel_sample_rate = mel_sample_rate

        self.sampling_attn_layer_idx = sampling_attn_layer_idx
        self.sampling_attn_head_idx = sampling_attn_head_idx
        self.sampling_attn_threshold = sampling_attn_threshold

        self.layers = layers
        self.model_dim = model_dim
        self.mlp_mult = mlp_mult

        self.voice_encoder = ConditioningEncoder(mel_dim, model_dim, voice_encoder_depth)
        self.voice_encoder_gain = nn.Parameter(torch.full((1, 1, model_dim), fill_value=0.02))

        # Parameters and settings for unconditional masking
        if version == 'v2':
            self.uncond_style = nn.Parameter(torch.randn(1, 4, model_dim))
        else:
            assert version == 'v3'
            self.uncond_style = nn.Parameter(torch.randn(1, 1, model_dim))
        self.uncond_speech = nn.Parameter(torch.randn(1, 1, model_dim))

        self.version = version

    @property
    def device(self):
        return self.voice_encoder_gain.device

    @property
    def dtype(self):
        return self.voice_encoder_gain.dtype
