import os
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device passed to google_qec_simulator")
    parser.add_argument("--shots", type=int, default=10000,
                        help="Monte-Carlo shots per circuit")
    args = parser.parse_args()
    # Get the directory of the currently executing script (so it's portable)
    script_dir = Path(__file__).resolve().parent  # this will work whether on local machine or in Google Drive
    print(f"Script is running from: {script_dir}")

    # Dynamically construct paths based on the current directory
    data_dir = script_dir / 'simulated_data'  # dataset directory (relative to script)
    output_dir = script_dir / 'simulated_data'  # output directory (relative to script)
    
    # Ensure the output directory exists
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Now we use the dynamically constructed paths to access files
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Example command to simulate: Running simulation for each file in data_dir
    npz_files = [f for f in data_dir.iterdir() if f.suffix == '.npz']
    print(f"Found {len(npz_files)} .npz files. Running simulations...")

    for idx, npz_file in enumerate(npz_files, start=1):
        print(f"=== Running simulation {idx}/{len(npz_files)} ===")
        print(f"File: {npz_file}")

        # Simulation command (example, adjust as needed)
        command = (
            f"python ./google_qec_simulator/main.py {npz_file} "
            f"--shots {args.shots} --device {args.device}"
        )
        print(f"Command: {command}")
        
        # Run your simulation here (subprocess or another method)
        # subprocess.run(command, shell=True)  # Uncomment this when you're ready to run
        
        # After running the simulation, load the .npz file from output directory
        output_file = output_dir / f"samples{npz_file.stem[7:]}.npz"  # Remove the 'samples_' prefix
        print(f"Output: Trying to load .npz file from {output_file}")
        
        if output_file.exists():
            print(f"Simulation completed successfully: {output_file}")
        else:
            print(f"Error: The file {output_file} does not exist.")

if __name__ == "__main__":
    main()
