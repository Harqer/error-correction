from datetime import datetime
import argparse
import os
import yaml
import numpy as np
from simulator.dem_generator import generate_dem_data
from simulator.si1000_generator import si1000_noise_model
from simulator.pauli_plus_simulator import PauliPlusSimulator
from google_qec_paper_noise_model.circuit_builder import SurfaceCodeCircuitBuilder
from google_qec_paper_noise_model.iq_readout import IQReadoutModel


def main(model_type: str, num_samples: int, basis: str):
    # Load configuration
    config_path = os.path.join("configs", f"{model_type}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Generate data based on model type
    if model_type == "dem":
        syndromes, logicals = generate_dem_data(num_samples, config)
    elif model_type == "si1000":
        circuit = si1000_noise_model(config)
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    elif model_type == "pauli_plus":
        sim = PauliPlusSimulator(config, basis)
        sampler = sim.circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)
    elif model_type == "paper_aligned":
        builder = SurfaceCodeCircuitBuilder(
            distance=config["distance"],
            rounds=config["rounds"],
            basis=config.get("basis", basis),
            processor=config.get("processor", "72_qubit_paper_aligned"),
        )
        circuit = builder.build_circuit()
        sampler = circuit.compile_detector_sampler()
        syndromes, logicals = sampler.sample(num_samples, separate_observables=True)

        # --- Soft I/Q (paper’s “soft inputs”) ---
        # Build a simple per-measurement I/Q posterior array for each sample.
        # We derive #measurements from the circuit (count M instructions once).
        num_meas = sum(1 for inst in circuit if getattr(inst, "name", "") == "M")
        iq_model: IQReadoutModel = builder.iq_model
        rng = np.random.default_rng(0)
        # Sample scalar I values and compute posteriors [p0,p1,pl] per measurement.
        # In practice one would tie this to the true state just before measurement;
        # here we provide statistically consistent soft inputs as in Methods.
        xs = rng.normal(loc=0.0, scale=1.0, size=(num_samples, num_meas))
        post = np.stack([
            iq_model.soft_vector(xs[i, j]) for i in range(num_samples) for j in range(num_meas)
        ], axis=0).reshape(num_samples, num_meas, 3)
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
    if model_type == "paper_aligned":
        iq_path = os.path.join(output_dir, f"{model_type}_iq_soft_{basis}_{timestamp}.npy")
        np.save(iq_path, post)

    print(f"Successfully generated {num_samples} samples")
    print(f"Syndromes saved to: {syndrome_path}")
    print(f"Logical errors saved to: {logicals_path}")
    if model_type == "paper_aligned":
        print(f"Soft I/Q posteriors saved to: {iq_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quantum error correction data")
    parser.add_argument("--model", choices=["dem", "si1000", "pauli_plus", "paper_aligned"], required=True,
                        help="Type of noise model to generate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--basis", type=str, default='z', help="basis:x or z")
    args = parser.parse_args()
    main(args.model, args.samples, args.basis)

