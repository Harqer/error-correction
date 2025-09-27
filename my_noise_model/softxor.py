"""softxor.py —— 软检测器事件的概率组合工具。"""

import numpy as np
from typing import Iterable, Tuple


def soft_xor(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """软 XOR：``p + q - 2pq``，支持向量化运算。"""
    return p + q - 2.0 * p * q

def soft_detection_sequence(meas_probs: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """根据每轮测量的软概率生成检测事件与泄漏轨迹。"""
    probs = [np.asarray(x, dtype=float) for x in meas_probs]

    if not probs:
        empty = np.zeros((0,), dtype=float)
        return empty, empty

    p1s = [p[..., 1] for p in probs]
    det_outs = [soft_xor(p1s[t - 1], p1s[t]) for t in range(1, len(p1s))]

    if det_outs:
        det = np.stack(det_outs, axis=0)
    else:
        det = np.zeros((0, *p1s[0].shape), dtype=float)

    leak = np.stack([p[..., 2] for p in probs], axis=0)
    return det, leak

