"""
    Quantized version of the Class Attention Block (pre-LN):
    x^{l} = x^{l-1} + Attn(LN(x^{l-1}), U)
    x^{l} = x^{l}   + MLP(LN(x^{l}))
    - P-MHA + MLP linears quantized
    - LayerNorm, GELU, Softmax in FP32
"""
import torch
import torch.nn as nn
from ..layers import *


class QuantClassAttentionBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0 # expansion ratio for MLP hidden dim
                 ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # 1) Pre-LN before MHA (float)
        self.ln1 = nn.LayerNorm(dim)

        # 2) Quantized Particle MHA
        self.mha = QMHALayer(embed_dim=dim,
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

    def forward(self, x_class, x_l, attn_mask=None):
        """
        x_class: [B, 1, D]   particle features (x^{l-1})
        x_l: [B, N, D]   particle features (x^{l-1})
        attn_mask: optional mask for attention (broadcastable to [B, H, N, N])

        returns: x^{l}  (same shape as x)
        """

        # ----- concat -----
        x_cat = torch.cat([x_class, x_l], dim=1)  # [B,N+1,D]
        attn_in = self.ln1(x_cat)          

        # ----- MHA -----
        attn_out = self.mha(attn_in)

        # take only class output
        xc_mha = attn_out[:, :1, :]
        xc_mha_norm = self.ln2(xc_mha)

        # ----- first residual -----
        y = self.res_add1(x_class, xc_mha_norm)

        # ----- MLP -----
        y_norm = self.ln3(y)
        h = self.fc1(y_norm)
        h = self.activation(h)
        h_norm = self.ln4(h)
        h = self.fc2(h_norm)

        # ----- second residual -----
        out = self.res2(y, h)
        return out

