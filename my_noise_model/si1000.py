"""si1000.py —— SI1000 电路去极化模型的权重生成器。"""

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class SI1000Weights:
    """记录 SI1000 模型中各类噪声的权重系数。"""

    meas_bitflip: float   # => 5p
    reset_bitflip: float  # => 2p
    resonator_idle: float # => 2p
    twoq_depol: float     # => p
    oneq_depol: float     # => p/10
    idle: float           # => p/10


def make_si1000_weights(p: float) -> Dict[str, float]:
    """根据基准误差 ``p`` 返回 SI1000 噪声权重。"""
    if p < 0:
        raise ValueError("p must be >= 0")
    return {
        "meas_bitflip": 5.0 * p,
        "reset_bitflip": 2.0 * p,
        "resonator_idle": 2.0 * p,
        "twoq_depol": 1.0 * p,
        "oneq_depol": 0.1 * p,  # p/10
        "idle": 0.1 * p,        # p/10
    }

