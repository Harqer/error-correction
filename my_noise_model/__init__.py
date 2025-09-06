"""
Paper-aligned noise builders for ALPHAQUBIT.
Exports:
  - make_si1000_weights
  - IQReadoutModel (full & simplified)
  - soft_xor, soft_detection_sequence
  - twirl_to_pauli_channel (GPTA)
  - build_pauli_plus_channels (leakage, crosstalk, DQLR, etc.)
"""
from .si1000 import make_si1000_weights
from .iq_readout import IQReadoutModel
from .softxor import soft_xor, soft_detection_sequence
from .gpta import twirl_to_pauli_channel
from .pauli_plus import build_pauli_plus_channels

__all__ = [
    "make_si1000_weights",
    "IQReadoutModel",
    "soft_xor",
    "soft_detection_sequence",
    "twirl_to_pauli_channel",
    "build_pauli_plus_channels",
]

