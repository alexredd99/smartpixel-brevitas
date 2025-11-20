import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity

class QResAddLayer(nn.Module):

    """
    Quantized residual add layer for QAT, matching AIE ResAddLayer behavior.

    Computes:
        out = x + y

    Then applies activation quantization (fake quant) via QuantIdentity
    to simulate int8 behavior on the residual output.

    Args:
        a_bit_width : bit width for activation quantization (default 8)

    Usage:
        out = QResAdd()(x, y)
    """
    def __init__(self, a_bit_width: int = 8):
        super().__init__()
        
        # Quantize the output activation after addition
        self.quant = QuantIdentity(bit_width=a_bit_width)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x and y should have the same shape: (B, T, d_model)
        z = x + y          # residual add (float add during QAT)
        z = self.quant(z)  # quantized activation (int8-like)
        return z
