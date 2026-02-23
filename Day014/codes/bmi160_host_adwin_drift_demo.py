import numpy as np
import matplotlib.pyplot as plt
from river import drift
import random


def generate_synthetic_stream(n_stable=1000, n_drift=1000):
    """
    Generates a stream of numbers with a sudden concept drift.
    - Phase 1: Random numbers between 0.0 and 1.0 (Mean ~0.5)
    - Phase 2: Random numbers between 2.0 and 3.0 (Mean ~2.5)
    """
    data_phase_1 = [random.uniform(0, 1) for _ in range(n_stable)]
    data_phase_2 = [random.uniform(2, 3) for _ in range(n_drift)]
    return data_phase_1 + data_phase_2


def main():
    # 1. Setup the Data Stream
    stream = generate_synthetic_stream()

    # 2. Initialize ADWIN Drift Detector
    # delta=0.002 is the confidence value (sensitivity)
    adwin = drift.ADWIN(delta=0.002)

    # For plotting
    plot_data = []
    drift_indices = []
    running_means = []

    print(f"Processing {len(stream)} samples...")

    # 3. Process the Stream
    for i, value in enumerate(stream):
        # Update the drift detector with the new value
        adwin.update(value)

        # Store data for visualization
        plot_data.append(value)
        running_means.append(adwin.estimation)  # ADWIN's internal estimate of the mean

        # Check if ADWIN detected a change
        if adwin.drift_detected:
            print(f"⚠️ Drift Detected at index: {i} | Input Value: {value:.4f}")
            drift_indices.append(i)
            # Note: ADWIN automatically resets its window after detection

    # 4. Visualization
    plt.figure(figsize=(12, 6))

    # Plot raw stream
    plt.plot(plot_data, label='Stream Data', alpha=0.5, color='gray')

    # Plot ADWIN's internal mean estimate
    plt.plot(running_means, label='ADWIN Mean Estimate', color='blue', linewidth=2)

    # Plot Drift Points
    for idx in drift_indices:
        plt.axvline(x=idx, color='red', linestyle='--', linewidth=2, label='Drift Detected')

    plt.title("Concept Drift Detection using River (ADWIN)")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    main()