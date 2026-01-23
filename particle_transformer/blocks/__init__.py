"""
Quantized block implementations using Brevitas.

"""

from .class_attention import QuantClassAttentionBlock
from .particle_attention import QuantParticleAttentionBlock

__all__ = ['QuantClassAttentionBlock', 'QuantParticleAttentionBlock']