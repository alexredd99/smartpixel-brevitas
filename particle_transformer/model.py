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
from layers import QDenseLayer, QMHALayer, QResAddLayer


class ParTModel:

    def __init__(
        self,
        in_features: int = 8,    # num_feature_pad
        d_model: int = 64,       # ff_dim
        num_heads: int = 4,
        out_dim: int = 8,
        w_bit_width: int = 8,
        a_bit_width: int = 8,
    ):
        super().__init__()

        # First dense to go from padded features → d_model
        self.dense0 = QDenseLayer(
            in_features,
            d_model,
            act=True,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )

        # First attention + residual
        self.mha1 = QMHALayer(
            d_model=d_model,
            num_heads=num_heads,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
            out_act=False,   # no ReLU after attention proj
        )
        self.res1 = QResAddLayer(a_bit_width=a_bit_width)

        # First FFN block (two dense layers) + residual
        self.ff1a = QDenseLayer(
            d_model,
            d_model,
            act=True,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )
        self.ff1b = QDenseLayer(
            d_model,
            d_model,
            act=True,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )
        self.res2 = QResAddLayer(a_bit_width=a_bit_width)

        # Second attention + residual
        self.mha2 = QMHALayer(
            d_model=d_model,
            num_heads=num_heads,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
            out_act=False,
        )
        self.res3 = QResAddLayer(a_bit_width=a_bit_width)

        # Second FFN block + residual
        self.ff2a = QDenseLayer(
            d_model,
            d_model,
            act=True,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )
        self.ff2b = QDenseLayer(
            d_model,
            d_model,
            act=True,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )
        self.res4 = QResAddLayer(a_bit_width=a_bit_width)

        # Output head
        self.out1 = QDenseLayer(
            d_model,
            d_model,
            act=True,
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )
        self.out2 = QDenseLayer(
            d_model,
            out_dim,
            act=False,   # final logits: no ReLU
            w_bit_width=w_bit_width,
            a_bit_width=a_bit_width,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, in_features)
        mask (optional): (B, T) with 1=keep, 0=pad
        """
        # dense0
        x0 = self.dense0(x)           # (B, T, d_model)

        # MHA 1 + residual
        a1 = self.mha1(x0, mask)
        x1 = self.res1(x0, a1)

        # FFN 1 + residual
        f1 = self.ff1b(self.ff1a(x1))
        x2 = self.res2(x1, f1)

        # MHA 2 + residual
        a2 = self.mha2(x2, mask)
        x3 = self.res3(x2, a2)

        # FFN 2 + residual
        f2 = self.ff2b(self.ff2a(x3))
        x4 = self.res4(x3, f2)

        # Output head
        h = self.out1(x4)
        out = self.out2(h)            # (B, T, out_dim)

        return out