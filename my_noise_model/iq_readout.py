import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)

@dataclass
class IQReadoutModel:
    """
    Implements the paper's 1D I/Q readout likelihoods and posteriors.
    - SNR sets separation of |0> and |1> Gaussian means (mu = SNR/2).
    - tau = t/T1 controls amplitude-damping that collapses |1| toward |0|.
    - Leakage 'L' uses a broad, near-centered distribution to reflect loss of contrast.
    - Returns *soft posteriors* P(s | x) for s in {0,1,L}.
    - Provides soft detection-event conversion via SoftXOR (see softxor.py).

    This follows the Methods description (soft inputs for measurements and detection events).
    """
    snr: float
    tau: float
    # Prior probability of leakage for the measurement under consideration.
    p_leak_prior: float = 1e-3
    sigma: float = 1.0
    leak_sigma_scale: float = 1.6

    def _means(self) -> Tuple[float, float]:
        mu = 0.5 * self.snr
        # amplitude damping collapses the |1> cloud toward |0|
        # We use alpha = exp(-tau) as the retained |1| amplitude component.
        alpha = math.exp(-self.tau)
        mu0 = +mu
        mu1 = -alpha * mu
        return mu0, mu1

    def posteriors(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Return dict with keys 'p0','p1','pl' as posteriors given scalar or array x."""
        x = np.asarray(x, dtype=float)
        mu0, mu1 = self._means()
        s = self.sigma
        sL = self.leak_sigma_scale * s
        # Priors: split non-leakage evenly between 0/1
        piL = float(self.p_leak_prior)
        pi0 = 0.5 * (1.0 - piL)
        pi1 = 0.5 * (1.0 - piL)
        L0 = _gauss_pdf(x, mu0, s)
        L1 = _gauss_pdf(x, mu1, s)
        LL = _gauss_pdf(x, 0.0, sL)
        num0 = pi0 * L0
        num1 = pi1 * L1
        numL = piL * LL
        Z = num0 + num1 + numL + 1e-18
        return {
            "p0": num0 / Z,
            "p1": num1 / Z,
            "pl": numL / Z,
        }

    def soft_meas_probs(self, x: np.ndarray) -> np.ndarray:
        """Return soft measurement probabilities [p0, p1, pl] for each sample."""
        post = self.posteriors(x)
        return np.stack([post["p0"], post["p1"], post["pl"]], axis=-1)

    def soft_meas_prob1(self, x: np.ndarray) -> np.ndarray:
        """Backward-compatibility helper returning only P(m=1)."""
        return self.soft_meas_probs(x)[..., 1]

    def soft_vector(self, x: np.ndarray) -> np.ndarray:
        """Alias of soft_meas_probs for historic callers."""
        return self.soft_meas_probs(x)

    # Convenience generators for synthetic I/Q samples (useful for tests)
    def sample_states(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw (state_labels, samples) where state_labels in {0,1,2} (2=leak).
        Priors match those used in 'posteriors'.
        """
        piL = float(self.p_leak_prior)
        pi0 = 0.5 * (1.0 - piL)
        pi1 = 0.5 * (1.0 - piL)
        probs = np.array([pi0, pi1, piL], dtype=float)
        states = rng.choice(3, size=n, p=probs)
        mu0, mu1 = self._means()
        mus = np.array([mu0, mu1, 0.0], dtype=float)
        sigs = np.array([self.sigma, self.sigma, self.leak_sigma_scale * self.sigma], dtype=float)
        xs = rng.normal(loc=mus[states], scale=sigs[states])
        return states, xs

