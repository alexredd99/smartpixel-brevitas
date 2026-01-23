"""
    QAT version of the ParT model:

        dense0 → mha1 → res1 →
        ff1a(dense) → ff1b(dense) → res2 →
        mha2 → res3 →
        ff2a(dense) → ff2b(dense) → res4 →
        out1(dense) → out2(dense)

    Assumes input shape: (B, T, in_features)
    where you can pad T up to num_particles_pad (e.g., 160) and in_features up to num_feature_pad (e.g., 8).
"""

import torch
from .blocks import QuantParticleAttentionBlock, QuantClassAttentionBlock
from .layers import QDenseLayer
import torch.nn as nn


class ParTModel(torch.nn.Module):

    def __init__(
        self,
        in_features: int = 8,    
        d_model: int = 128,      
        num_heads: int = 8,
        num_classes: int = 10,
        w_bit_width: int = 8,
        a_bit_width: int = 8,
        pab_num = 8,       # number of Particle Attention Blocks
        cab_num = 2,       # number of Class Attention Blocks
    ):
        super().__init__()

        self.pab_num = pab_num
        self.cab_num = cab_num

        self.pab_blocks = torch.nn.ModuleList([
            QuantParticleAttentionBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=4.0,
            )
            for _ in range(pab_num)
        ])

        self.cab_blocks = torch.nn.ModuleList([
            QuantClassAttentionBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=4.0,
            )
            for _ in range(cab_num)
        ])

        # single-layer MLP, followed by softmax
        self.mlp = QDenseLayer(in_features=d_model, out_features=num_classes, act=False, w_bit_width=w_bit_width, a_bit_width=a_bit_width)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x: torch.Tensor, U: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, in_features)
        mask (optional): (B, T) with 1=keep, 0=pad
        """
        # Particle Attention Blocks
        for _ in range(self.pab_num):
            x = self.pab_blocks[_](x, U, mask)

        # Initial input for Class Attention Blocks
        x_pab_out = x # save PAB output for CAB input
        x_class = x[:, :1, :]  # class token
        
        # Class Attention Blocks
        for _ in range(self.cab_num):
            x_l = x_pab_out     # particle tokens
            x_class = self.cab_blocks[_](x_class, x_l, mask)

        out = self.mlp(x_class)  # [B, 1, num_classes]
        out = self.softmax(out)   # [B, 1, num_classes]
        
        return out