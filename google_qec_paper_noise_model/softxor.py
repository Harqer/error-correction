import numpy as np
from typing import Iterable, Tuple

def soft_xor(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Soft XOR for Bernoulli probabilities:
        XOR(p, q) = p + q - 2 p q
    Works element-wise for arrays.
    """
    return p + q - 2.0 * p * q

def soft_detection_sequence(meas_probs: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a time-ordered iterable of soft measurement probabilities with final
    axis [p0, p1, pl], return a tuple `(det_probs, leak_probs)` where:

      * `det_probs[t]` is the soft detection-event probability between rounds
        `t` and `t-1`, computed from the P(m=1) components via soft XOR.
      * `leak_probs[t]` is the marginal probability of leakage at round `t`.

    Shapes:
      - det_probs: (#rounds-1, ...)
      - leak_probs: (#rounds, ...)
    """
    probs = [np.asarray(x, dtype=float) for x in meas_probs]
    p1s = [p[..., 1] for p in probs]
    det_outs = []
    for t in range(1, len(p1s)):
        det_outs.append(soft_xor(p1s[t - 1], p1s[t]))
    det = np.stack(det_outs, axis=0)
    leak = np.stack([p[..., 2] for p in probs], axis=0)
    return det, leak

