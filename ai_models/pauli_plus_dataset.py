import torch
from torch.utils.data import Dataset
import numpy as np
import math
import os
import logging
 

# Set up logging
logging.basicConfig(level=logging.INFO)  # Change this to logging.DEBUG for debug mode
logger = logging.getLogger()

# 1. Helper function to load and reshape syndromes
def load_and_reshape_syndromes(syndromes):
    logger.debug(f"Original syndromes shape: {syndromes.shape}")
    if syndromes.ndim == 2:
        syndromes = syndromes[:, :, None, None]  # Reshape to (N, S, 1, 1)
        logger.debug(f"Reshaped syndromes to: {syndromes.shape}")
        return syndromes, 1, 1  # Assuming 1 round, 1 feature
    elif syndromes.ndim == 3:
        return syndromes, *syndromes.shape[1:]  # Return R, S, F
    else:
        raise ValueError(f"Unexpected syndromes shape: {syndromes.shape}")

# 2. Helper function to concatenate the basis feature
def concatenate_basis_feature(syndromes, basis_id):
    basis_feat = np.full((syndromes.shape[0], syndromes.shape[1], syndromes.shape[2], 1), basis_id, dtype=syndromes.dtype)
    logger.debug(f"Basis feature tensor shape: {basis_feat.shape}")
    concatenated = torch.tensor(np.concatenate([syndromes, basis_feat], axis=-1), dtype=torch.float32)
    logger.debug(f"Shape after concatenation: {concatenated.shape}")
    return concatenated

# 3. Helper function to calculate padding
def calculate_padding(S):
    d = math.isqrt(S + 1)
    if d * d != S + 1:
        logger.debug(f"Padding needed: Current stabilizer count = {S}")
        d += 1
        pad = d * d - 1 - S
        logger.debug(f"Calculated padding: {pad}")
        return pad
    else:
        return 0

# 4. Helper function to apply padding to syndromes
def apply_padding_to_syndromes(syndromes, pad):
    if pad > 0:
        logger.debug(f"Applying padding of {pad} to stabilizer dimension.")
        padded_syndromes = torch.cat([syndromes, torch.zeros((syndromes.shape[0], syndromes.shape[1], pad, syndromes.shape[3]), dtype=syndromes.dtype)], axis=2)
        logger.debug(f"Shape after padding: {padded_syndromes.shape}")
        return padded_syndromes
    return syndromes

# 5. Helper function to apply padding to the final mask
def apply_padding_to_mask(final_mask, pad, num_samples):
    if pad > 0:
        logger.debug(f"Padding final mask by {pad}.")
        padded_mask = torch.cat([final_mask, torch.zeros((num_samples, pad), dtype=final_mask.dtype)], axis=1)
        logger.debug(f"Shape of final mask after padding: {padded_mask.shape}")
        return padded_mask
    return final_mask

# 6. Helper function to update the final mask
def update_final_mask(S, final_mask, d):
    logger.debug(f"Final stabilizer count after padding: {S}")
    for idx in range(S):
        r, c = idx // d, idx % d
        logger.debug(f"Updating final_mask at index {idx}, (r, c) = ({r}, {c})")
        final_mask[:, idx] = 1 if (r + c) % 2 == 0 else 2
        logger.debug(f"Updated final_mask at index {idx}: {final_mask[:5]}")  # Show a preview of the updated final mask
    return final_mask


# 7. Main dataset class with individual functions
# ai_models/pauli_plus_dataset.py



class PauliPlusDataset(Dataset):
    """
    Loads one NPZ produced by google_qec_simulator and returns
        (inputs, basis_id, final_mask) , label
    where
        inputs      : float32 [R,S,3]
        basis_id    : 0 = X, 1 = Z, -1 = unknown / mixed
        final_mask  : int8   [S]   (0 bulk, 1 on-basis final, 2 off-basis final)
        label       : float32 scalar (logical error 0/1)
    """
    def __init__(self, npz_path: str, basis_id: int):
        super().__init__()
        self.npz_path  = npz_path
        self.basis_id  = basis_id

        data = np.load(npz_path)
        self.x = torch.tensor(data["data"], dtype=torch.float32)   # (N,R,S,3)
        num_samples = self.x.shape[0]

        # ----------------------------------------------------------
        # choose label array with a robust priority order
        # ----------------------------------------------------------

        def _register_candidate(keys, bucket):
            for key in keys:
                if key in data.files and key not in bucket:
                    bucket.append(key)

        def _is_valid(candidate, name):
            if candidate.ndim == 0:
                logger.warning(
                    "Skipping '%s' from %s because it is a scalar", name, npz_path
                )
                return False
            if candidate.shape[0] != num_samples:
                logger.warning(
                    "Skipping '%s' from %s because first dimension %s does not match %s samples",
                    name,
                    npz_path,
                    candidate.shape[0],
                    num_samples,
                )
                return False
            # collapse trailing dimensions to detect empty feature axes
            if candidate.ndim > 1:
                trailing = int(np.prod(candidate.shape[1:]))
                if trailing == 0:
                    logger.warning(
                        "Skipping '%s' from %s because it has zero columns", name, npz_path
                    )
                    return False
            return True

        label_candidates = []

        if self.basis_id == 0:  # X-basis
            _register_candidate(["obs_x", "logical_x"], label_candidates)
        elif self.basis_id == 1:  # Z-basis
            _register_candidate(["obs_z", "logical_z"], label_candidates)
        else:  # Unknown basis
            _register_candidate(["obs_x", "obs_z", "logical_x", "logical_z"], label_candidates)

        _register_candidate(["obs", "label", "labels", "logical", "observables"], label_candidates)
        _register_candidate([k for k in data.files if k != "data"], label_candidates)

        label_key = None
        for cand in label_candidates:
            candidate = data[cand]
            if not _is_valid(candidate, cand):
                continue
            label_key = cand
            break

        if label_key is None:
            raise ValueError(
                f"Could not find a valid label array in {npz_path}. Available keys: {data.files}"
            )

        y = data[label_key]
        if y.ndim > 1:
            y = y.reshape(num_samples, -1)
            y = y[:, 0]

        self.y = torch.tensor(y, dtype=torch.float32)


        # final‐round mask (0 bulk, 1 on-basis stabilizer, 2 off-basis stabilizer)
        R, S = self.x.shape[1:3]
        mask = np.zeros(S, dtype=np.int8)
        # first (S//2) are 'on' and the rest 'off' in rotated surface codes
        mask[0 : S // 2] = 1
        mask[S // 2 :]   = 2
        self.final_mask = torch.tensor(mask, dtype=torch.int8)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        xb   = self.x[idx]           # (R,S,3)
        mask = self.final_mask       # (S,)
        basis= torch.tensor(self.basis_id, dtype=torch.int8)
        yb   = self.y[idx]           # scalar
        return (xb, basis, mask), yb

# Main function to test the dataset class and each step
def main():
    # For testing, make sure to use an actual .npz file in your system.
    npz_file = './output/samples.npz'  # Example path, modify if needed
    print(f"Trying to load .npz file from {npz_file}")

    # Check if the file exists
    if not os.path.exists(npz_file):
        print(f"Error: The file {npz_file} does not exist.")
        return

    basis_id = 0  # Example basis ID for testing
    
    # Step 1: Test reshaping of syndromes
    syndromes = np.load(npz_file)['data']
    syndromes, R, F = load_and_reshape_syndromes(syndromes)
    logger.debug(f"Reshaped syndromes: {syndromes.shape}")

    # Step 2: Test concatenating basis feature tensor
    X = concatenate_basis_feature(syndromes, basis_id)
    logger.debug(f"Shape of X after concatenation: {X.shape}")

    # Step 3: Test padding calculation
    pad = calculate_padding(syndromes.shape[2])
    logger.debug(f"Padding calculated: {pad}")

    # Step 4: Test final mask update
    final_mask = torch.zeros(syndromes.shape[0], 3, dtype=torch.long)  # Adjust for the correct final mask size
    final_mask = update_final_mask(syndromes.shape[2] + pad, final_mask, math.isqrt(syndromes.shape[2] + pad + 1))
    logger.debug(f"Final mask after update: {final_mask[:5]}")

    # Step 5: Test apply padding to mask
    final_mask = apply_padding_to_mask(final_mask, pad, syndromes.shape[0])
    logger.debug(f"Final mask after applying padding: {final_mask[:5]}")

    # Step 6: Create the dataset object
    dataset = PauliPlusDataset(npz_file, basis_id)
 
 

# Only run the main function if this file is executed directly
if __name__ == '__main__':
    main()
