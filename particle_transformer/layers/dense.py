import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity

class QDenseLayer(nn.Module):

    """
    Quantized dense layer using Brevitas.

    Args:
        in_features   : input dimension
        out_features  : output dimension
        bias          : use bias or not
        act           : if True, apply quantized ReLU after linear
        w_bit_width   : weight bit width (default 8)
        a_bit_width   : activation bit width (default 8)

    This roughly corresponds to AIE DenseLayer:
        - Linear with int8 weights (via quantization)
        - Optional ReLU (is_relu flag)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act: bool = True,
        w_bit_width: int = 8,
        a_bit_width: int = 8,
    ):
        super().__init__()

        # Quantized linear layer
        self.linear = QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            weight_bit_width=w_bit_width,
            weight_per_channel=True,
        )

        # Quantized activation (or identity)
        if act:
            self.activation = QuantReLU(bit_width=a_bit_width, inplace=False)
        else:
            # still uses a quantizer but without identity nonlinearity
            self.activation = QuantIdentity(return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Brevitas expects float tensors during QAT; it will insert fake quant.
        x = self.linear(x)
        x = self.activation(x)
        return x