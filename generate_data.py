from datetime import datetime
import argparse
import os
import yaml
import numpy as np
from simulator.dem_generator import generate_dem_data
from simulator.si1000_generator import si1000_noise_model
from simulator.pauli_plus_simulator import PauliPlusSimulator


def main(model_type: str, num_samples: int, basis: str):
    # Load configuration
    config_path = os.path.join("configs", f"{model_type}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Generate data based on model type
    if model_type == "dem":
        syndromes, logicals = generate_dem_data(num_samples, config)
    elif model_type == "si1000":
        circuit = si1000_noise_model(config["p"])
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    elif model_type == "pauli_plus":
        sim = PauliPlusSimulator(config, basis)
        sampler = sim.circuit.compile_detector_sampler()  # or however your class exposes it
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    elif model_type == "paper_aligned":
        # Delegate to the canonical paper implementation shipped in the repo.
        from simulator.paper_aligned_adapter import build_paper_aligned_circuit
        circuit = build_paper_aligned_circuit(config, basis)
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save data as numpy arrays
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    syndrome_path = os.path.join(output_dir, f"{model_type}_syndromes_{basis}_{timestamp}.npy")
    logicals_path = os.path.join(output_dir, f"{model_type}_logicals_{basis}_{timestamp}.npy")
    np.save(syndrome_path, syndromes)
    np.save(logicals_path, logicals)

    print(f"Successfully generated {num_samples} samples")
    print(f"Syndromes saved to: {syndrome_path}")
    print(f"Logical errors saved to: {logicals_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quantum error correction data")
    parser.add_argument("--model", choices=["dem", "si1000", "pauli_plus", "paper_aligned"], required=True,
                        help="Type of noise model to generate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--basis", type=str, default='z', help="basis:x or z")
    args = parser.parse_args()
    main(args.model, args.samples, args.basis)

