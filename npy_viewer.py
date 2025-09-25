# stats.py
import sys
from pathlib import Path

import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python npy_viewer.py <path_to_npy_file>")
        return

    file_path = sys.argv[1]
    path = Path(file_path)

    if not path.exists():
        print(f"Error: {file_path} does not exist.")
        parent = path.parent if path.parent != Path("") else Path.cwd() / "output"
        if parent.exists():
            npy_files = sorted(parent.glob("*.npy"))
            if npy_files:
                print("\nAvailable .npy files:")
                for candidate in npy_files:
                    print(f"  - {candidate}")
        return

    data = np.load(path)
    
    # Basic stats
    print(f"\n{' File Info ':-^40}")
    print(f"Path: {path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min/Mean/Max: {data.min()} / {data.mean():.2f} / {data.max()}")
    
    # Display first 100 entries with smart formatting
    print(f"\n{' First 100 Entries ':-^40}")
    flat_data = data.reshape(-1)[:100]  # Handle any dimensionality
    
    with np.printoptions(
        threshold=100, 
        edgeitems=5, 
        linewidth=120,
        formatter={'int': lambda x: f"{x:2d}"}  # Formatting for binary data
    ):
        if len(data.shape) == 1:
            print(flat_data)
        else:
            # For 2D+ data, show first 100 elements with original structure
            print(data[:100] if len(data) > 100 else data)

if __name__ == "__main__":
    main()