import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantLinear, QuantIdentity, QuantReLU
from layers import *


class QuantQKVProj(nn.Module):
    """
    Quantized QKV projection layer using Brevitas.
    Args:
        embed_dim : embedding dimension (D)
        use_bias  : whether to use bias in linear layers
    This layer projects input x to Q, K, V using quantized linear layers.
    It uses QuantLinear from Brevitas for quantization-aware training.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv = QDenseLayer(embed_dim, 3 * embed_dim)

        # Quantized activations for input and QKV output
        self.act_in = QuantIdentity()
        self.act_qkv = QuantIdentity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, N, D]
        x_q = self.act_in(x)
        qkv = self.qkv(x_q)            # [B, N, 3D]
        qkv = self.act_qkv(qkv)
        q, k, v = qkv.chunk(3, dim=-1) # each [B, N, D]
        return q, k, v


def attention_mask_handler(
        attention_mask, batch_size, num_heads, query_seq_length, key_value_seq_length):
    """Re-arrange attention mask to go from 4D to 3D (explicit batch_size and n_heads) or 2D
    (implicit batch_size and n_heads)."""
    if len(attention_mask.shape) == 4:
        if attention_mask.shape[0] == 1:
            attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)
        if attention_mask.shape[1] == 1:
            attention_mask = attention_mask.repeat(1, num_heads, 1, 1)
        if attention_mask.shape[2] == 1:
            attention_mask = attention_mask.repeat(1, 1, query_seq_length, 1)
        attention_mask = attention_mask.view(
            batch_size * num_heads, query_seq_length, key_value_seq_length)
    elif len(attention_mask.shape) == 2 and attention_mask.shape[0] == 1:
        # This could happen in Encoder-like architecture
        assert query_seq_length == key_value_seq_length
        attention_mask = attention_mask.repeat(query_seq_length, 1)
    return attention_mask

class QMHALayer(nn.Module):
    """
    Quantized Multi-Head Attention (MHA) layer using Brevitas. For class attention blocks.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        self.qkv_proj = QuantQKVProj(embed_dim)
        self.out_proj = QDenseLayer(embed_dim, embed_dim)

        self.act_attn_out = QuantIdentity()
        self.act_out = QuantIdentity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, H, N, Hd]
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, N, Hd] -> [B, N, D]
        B, H, N, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Hd)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:         [B, N, D]       particle features
        key_value: [B, N_kv, D] or None（usually equals to x）
        attn_mask: Optional attention mask，broadcastable to [B, H, N, N_kv]
        key_value here allows MHA use different key/value inputs
        """
        B, N, D = x.shape
        if key_value is None:
            key_value = x
        _, N_kv, _ = key_value.shape

        Q, _, _ = self.qkv_proj(x)          # Q: [B,N,D]
        _, K, V = self.qkv_proj(key_value)  # K,V: [B,N_kv,D]

        Qh = self._split_heads(Q)           # [B,H,N,Hd]
        Kh = self._split_heads(K)           # [B,H,N_kv,Hd]
        Vh = self._split_heads(V)           # [B,H,N_kv,Hd]

        # Ensure contiguous
        Qh = Qh.contiguous()
        Kh = Kh.contiguous()
        Vh = Vh.contiguous()

        # scores: [B,H,N,N_kv]
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale

        # mask
        if attn_mask is not None:
            mask = attention_mask_handler(
                attn_mask, batch_size=B, num_heads=self.num_heads,
                query_seq_length=N, key_value_seq_length=N_kv
            )  # [B,H,N,N_kv]
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        context = torch.matmul(attn, Vh)        # [B,H,N,Hd]
        context = self._merge_heads(context)    # [B,N,D]

        context_q = self.act_attn_out(context)
        out = self.out_proj(context_q)
        out = self.act_out(out)
        return out                              # [B,N,D]


class QPMHALayer(nn.Module):
    """
    Quantized Particle Multi-Head Attention (P-MHA) layer using Brevitas. For particle attention blocks.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        self.qkv_proj = QuantQKVProj(embed_dim)
        self.out_proj = QDenseLayer(embed_dim, embed_dim)

        self.act_attn_out = QuantIdentity()
        self.act_out = QuantIdentity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Hd)

    def _broadcast_U(
        self,
        U: torch.Tensor,
        B: int,
        Nq: int,
        Nk: int,
    ) -> torch.Tensor:
        """
        Broadcast U to shape [B, H, Nq, Nk] for addition to attention scores.
        U can have shape:
          - [Nq, Nk]
          - [B, Nq, Nk]
          - [1, Nq, Nk]
          - [B, 1, Nq, Nk]
          - [B, H, Nq, Nk]
        """
        u = U
        # 2D: [Nq, Nk]
        if u.dim() == 2:
            u = u.unsqueeze(0).unsqueeze(0)   # [1,1,Nq,Nk]
        elif u.dim() == 3:
            # [B, Nq, Nk] or [1, Nq, Nk]
            u = u.unsqueeze(1)                # [B,1,Nq,Nk]
        # 4D: [B or 1, H or 1, Nq, Nk]
        if u.size(0) == 1 and B > 1:
            u = u.repeat(B, 1, 1, 1)
        if u.size(1) == 1 and self.num_heads > 1:
            u = u.repeat(1, self.num_heads, 1, 1)
        # cut to correct size
        u = u[:, :, :Nq, :Nk]
        return u

    def forward(
        self,
        x: torch.Tensor,
        U: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:         [B, N, D]       particle features
        U:         interaction term，add to logits
        key_value: [B, N_kv, D] or None（usually equals to x）
        attn_mask: same to QuantMHA
        """
        B, N, D = x.shape
        if key_value is None:
            key_value = x
        _, N_kv, _ = key_value.shape

        Q, _, _ = self.qkv_proj(x)
        _, K, V = self.qkv_proj(key_value)

        Qh = self._split_heads(Q)            # [B,H,N,Hd]
        Kh = self._split_heads(K)            # [B,H,N_kv,Hd]
        Vh = self._split_heads(V)            # [B,H,N_kv,Hd]

        Qh = Qh.contiguous()
        Kh = Kh.contiguous()
        Vh = Vh.contiguous()

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale  # [B,H,N,N_kv]

        # ===== Add U term =====
        if U is not None:
            u = self._broadcast_U(U, B=B, Nq=N, Nk=N_kv)  # [B,H,N,N_kv]
            scores = scores + u
        # ============================

        if attn_mask is not None:
            mask = attention_mask_handler(
                attn_mask, batch_size=B, num_heads=self.num_heads,
                query_seq_length=N, key_value_seq_length=N_kv
            )
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        context = torch.matmul(attn, Vh)          # [B,H,N,Hd]
        context = self._merge_heads(context)      # [B,N,D]

        context_q = self.act_attn_out(context)
        out = self.out_proj(context_q)
        out = self.act_out(out)
        return out
