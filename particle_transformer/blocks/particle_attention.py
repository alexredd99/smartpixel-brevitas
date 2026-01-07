"""
    Quantized version of the Particle Attention Block (pre-LN):
    x^{l} = x^{l-1} + Attn(LN(x^{l-1}), U)
    x^{l} = x^{l}   + MLP(LN(x^{l}))
    - P-MHA + MLP linears quantized
    - LayerNorm, GELU, Softmax in FP32
"""
import torch
import torch.nn as nn
from particle_transformer.layers import *


class QuantParticleAttentionBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0 # expansion ratio for MLP hidden dim
                 ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # 1) Pre-LN before P-MHA (float)
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)

        # 2) Quantized Particle MHA
        self.pmha = QPMHALayer(embed_dim=dim,
                              num_heads=num_heads)
                              
        # 3) Residual-add after attention (quantized)
        self.res_add1 = QResAddLayer()

        # 4) Pre-LN before MLP (float)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.ln3 = nn.LayerNorm(dim, eps=1e-6)

        # 5) Quantized MLP: Dense -> GELU -> LN -> Dense
        self.fc1 = QDenseLayer(dim, hidden_dim)
        self.fc2 = QDenseLayer(hidden_dim, dim)

        self.activation = nn.GELU()
        self.ln4 = nn.LayerNorm(hidden_dim, eps=1e-6)

        # 6) Residual-add after MLP (quantized)
        self.res_add2 = QDenseLayer()

    def forward(self, x, U, attn_mask=None):
        """
        x: [B, N, D]   particle features (x^{l-1})
        U: [B, N, D_u] interaction embeddings for P-MHA
        attn_mask: optional mask for attention (broadcastable to [B, H, N, N])

        returns: x^{l}  (same shape as x)
        """

        # LN1 in float
        x_norm = self.ln1(x)

        # Quantized P-MHA
        attn_out = self.pmha(x_norm, U, attn_mask)   # [B, N, D]
        attn_out_norm = self.ln2(attn_out)

        # Quantized residual add: x + attn_out
        y = self.res_add1(x, attn_out_norm)

        # ------- MLP sub-block -------
        y_norm = self.ln3(y)           # LN3 in float

        # Quantized Dense -> GELU -> Quantized Dense
        h = self.fc1(y_norm)
        h = self.activation(h)         # GELU usually kept in float
        h_norm = self.ln4(h)         # LN4 in float
        h = self.fc2(h_norm)

        # Second residual: y + h
        out = self.res_add2(y, h)

        return out
