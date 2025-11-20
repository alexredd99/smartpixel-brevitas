import math
import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantIdentity, QuantReLU


class QMHALayer(nn.Module):
    """
    Quantized Multi-Head Attention layer using Brevitas.

    This corresponds to the MHA blocks in Particle Transformer:
        - Q = x Wq
        - K = x Wk
        - V = x Wv
        - Attention(Q, K, V)
        - Output projection Wo

    Args:
        d_model      : total embedding dimension
        num_heads    : number of attention heads
        bias         : whether to use bias in linear layers
        w_bit_width  : bit width for weights (default 8)
        a_bit_width  : bit width for activations (default 8)
        out_act      : if True, apply QuantReLU after out projection;
                       otherwise use QuantIdentity (quantized but no nonlinearity)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        w_bit_width: int = 8,
        a_bit_width: int = 8,
        out_act: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Quantized linear projections for Q, K, V
        self.q_proj = QuantLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
            weight_bit_width=w_bit_width,
            weight_per_channel=True,
        )
        self.k_proj = QuantLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
            weight_bit_width=w_bit_width,
            weight_per_channel=True,
        )
        self.v_proj = QuantLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
            weight_bit_width=w_bit_width,
            weight_per_channel=True,
        )

        # Output projection Wo
        self.out_proj = QuantLinear(
            in_features=d_model,
            out_features=d_model,
            bias=bias,
            weight_bit_width=w_bit_width,
            weight_per_channel=True,
        )

        # Quantized activation after Wo:
        # - usually identity (no nonlinearity) but still quantized
        if out_act:
            self.out_act = QuantReLU(bit_width=a_bit_width, inplace=False)
        else:
            self.out_act = QuantIdentity(bit_width=a_bit_width)

    def _shape(self, x: torch.Tensor, bsz: int):
        """
        Reshape (B, T, d_model) -> (B, num_heads, T, d_head)
        """
        return (
            x.view(bsz, -1, self.num_heads, self.d_head)
             .transpose(1, 2)  # (B, num_heads, T, d_head)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x   : input tensor of shape (B, T, d_model)
            mask: optional attention mask
                  shape (B, T) or (B, 1, 1, T) where 0 = masked, 1 = keep

        Returns:
            Tensor of shape (B, T, d_model)
        """
        B, T, _ = x.shape

        # Quantized projections (weights + activations quantized by Brevitas)
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = self._shape(q, B)  # (B, h, T, d_head)
        k = self._shape(k, B)  # (B, h, T, d_head)
        v = self._shape(v, B)  # (B, h, T, d_head)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, h, T, T)
        scores = scores / math.sqrt(self.d_head)

        # Apply mask if provided
        if mask is not None:
            # Accept (B, T) or (B, 1, 1, T)
            if mask.dim() == 2:
                mask_expanded = mask[:, None, None, :]  # (B,1,1,T)
            else:
                mask_expanded = mask
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, h, T, T)

        # Attention output
        context = torch.matmul(attn, v)       # (B, h, T, d_head)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)

        # Output projection Wo + quantized activation
        out = self.out_proj(context)          # (B, T, d_model)
        out = self.out_act(out)               # QuantIdentity or QuantReLU

        return out
