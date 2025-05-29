from typing import Optional
import torch
import torch.nn as nn
from torchtune.modules import FeedForward, RMSNorm
from playdiffusion.models.inpainter.position_embeddings import RotaryPositionalEmbeddings
from playdiffusion.models.inpainter.llm import TransformerDecoderLayer, NARSelfAttention, prepare_mask

class DiffLlama(nn.Module):
    def __init__(
        self,
        num_layers: int = 20,
        num_heads: int = 16,
        num_kv_heads: int = 16,
        embed_dim: int = 1024,
        intermediate_dim: int = 4096,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_base: float = 500000.0,
    ):
        super().__init__()

        # Parameters
        self.dim = embed_dim
        self.hidden_dim = intermediate_dim
        self.dim_head = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.attn_dropout = attn_dropout

        # Actual layers/parameters
        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.layers.append(self.build_layer())
        self.norm = RMSNorm(self.dim, eps=self.norm_eps)

    def build_layer(self):
        rope = RotaryPositionalEmbeddings(
            dim=self.dim_head, max_seq_len=self.max_seq_len, base=self.rope_base
        )

        self_attn = NARSelfAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.dim_head,
            q_proj=nn.Linear(self.dim, self.num_heads * self.dim_head, bias=False),
            k_proj=nn.Linear(self.dim, self.num_kv_heads * self.dim_head, bias=False),
            v_proj=nn.Linear(self.dim, self.num_kv_heads * self.dim_head, bias=False),
            output_proj=nn.Linear(self.dim, self.dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=self.max_seq_len,
            attn_dropout=self.attn_dropout,
        )
        mlp = FeedForward(
            gate_proj=nn.Linear(self.dim, self.hidden_dim, bias=False),
            down_proj=nn.Linear(self.hidden_dim, self.dim, bias=False),
            up_proj=nn.Linear(self.dim, self.hidden_dim, bias=False),
        )
        return TransformerDecoderLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=self.dim, eps=self.norm_eps),
            mlp_norm=RMSNorm(dim=self.dim, eps=self.norm_eps),
        )

    def forward(
        self,
        h,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Only outputs the latents, before passing it through the output layer
        h: [b, s, d]
        mask: b, s
        Returns:
            Tensor: output tensor with shape [b x s x d]
        """
        if mask is not None:      # B, T
            mask = prepare_mask(mask)       # B, 1, T, T

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask)

        # shape: [b, s, d]
        h = self.norm(h)
        return h
