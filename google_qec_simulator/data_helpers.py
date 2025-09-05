# google_qec_simulator/data_helpers.py
import numpy as np
import torch
from pathlib import Path
from stim_helpers import extract_rounds_and_dets


def reshape_detectors(det_flat: np.ndarray, stim_path: Path, shots: int) -> np.ndarray:
    """
    Reshapes the flat detector array into a (shots, rounds, dets_per_round, 1) shape.
    """
    # Get rounds and detectors per round
    R, S = extract_rounds_and_dets(stim_path)
    
    # Debugging: Print out rounds, detectors, and shot information
    print(f"Shots (N): {shots}")
    print(f"Extracted rounds (R): {R}, Detectors per round (S): {S}")
    
    # Calculate the expected reshape size
    expected_size = shots * R * S
    print(f"Expected reshape size (shots * rounds * detectors_per_round): {expected_size}")
    
    # Ensure the reshaping size matches
    print(f"Actual detected data size: {det_flat.size}")
    if det_flat.size != expected_size:
        raise ValueError(f"Cannot reshape {det_flat.size} elements into "
                         f"shape ({shots}, {R}, {S}, 1). "
                         f"Expected size: {expected_size}. Please check the dimensions of your data.")

    return det_flat.reshape(shots, R, S, 1).astype(np.float32)


def soft_channels(n, snr=10.0, t=0.01, leak_p=0.00275, device: str = "cpu"):
    """Vectorized soft-channel sampling that can run on CPU, GPU or NPU.

    Parameters
    ----------
    n : int
        Number of samples to draw.
    snr : float
        Signal-to-noise ratio of the readout channel.
    t : float
        Characteristic decay time used in the leakage model.
    leak_p : float
        Probability of preparing in the leakage state.
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda"``, ``"npu"``).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Posterior probabilities for states 1 and 2.
    """

    dev = torch.device(device)

    # Sample physical states {0,1,2}
    probs = torch.tensor([1 - leak_p - 0.5, 0.5, leak_p], device=dev)
    states = torch.multinomial(probs, n, replacement=True)

    # Sample IQ points with additive Gaussian noise and optional decay
    std = 1 / torch.sqrt(torch.tensor(snr, device=dev))
    z = torch.normal(states.float(), std)
    mask = states >= 1
    if mask.any():
        z[mask] -= t * torch.empty(mask.sum(), device=dev).exponential_()

    # Compute PDF values for the three hypotheses
    p0 = torch.exp(-snr * z**2)
    p1 = 0.5 * torch.exp(-snr * (z - 1) ** 2) + 0.5 * torch.exp(-snr * z**2) * torch.exp(-z / t) * (z > 0)
    p2 = 0.5 * torch.exp(-snr * (z - 2) ** 2) + 0.5 * torch.exp(-snr * (z - 1) ** 2) * torch.exp(-(z - 1) / (2 * t)) * (z > 1)

    # Convert PDFs to posterior probabilities with fixed priors
    w0, w1, w2 = 0.495, 0.495, 0.01
    norm = w0 * p0 + w1 * p1 + w2 * p2
    post1 = torch.where(norm == 0, torch.full_like(norm, 0.5), (w1 * p1) / norm)
    post2 = torch.where(norm == 0, torch.zeros_like(norm), (w2 * p2) / norm)

    return post1.cpu().numpy().astype(np.float32), post2.cpu().numpy().astype(np.float32)






# Test each function in this file
if __name__ == "__main__":
    stim_path = Path("path_to_a_stim_file/stim_example.stim")

    # Test reshape_detectors
    det_flat = np.random.randint(0, 2, (1000, 100))  # Fake data
    reshaped_det = reshape_detectors(det_flat, stim_path)
    print(f"Reshaped detector shape: {reshaped_det.shape}")

    # Test soft channels generation
    post1, post2 = soft_channels(1000)
    print(f"Generated post1 and post2 channels shapes: {post1.shape}, {post2.shape}")
