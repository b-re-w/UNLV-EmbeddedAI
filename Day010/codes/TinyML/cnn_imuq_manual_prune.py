import tensorflow as tf
import numpy as np
import os

# Configuration
INPUT_MODEL = "cnn_imu.h5"
OUTPUT_MODEL = "cnn_imu_pruned_manual.h5"
THRESHOLD = 0.05  # Weights smaller than 0.05 will be set to 0.0

print(f"Loading {INPUT_MODEL}...")
model = tf.keras.models.load_model(INPUT_MODEL)

print("Applying Manual Magnitude Pruning...")
total_weights = 0
pruned_weights = 0

# Iterate through all layers
for layer in model.layers:
    # Only prune layers with weights (Dense, Conv1D)
    if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
        weights = layer.get_weights()
        new_weights = []

        for w in weights:
            # Create a mask where condition is met (abs(w) < threshold)
            mask = np.abs(w) > THRESHOLD

            # Apply mask: Keep large weights, zero out small ones
            w_pruned = w * mask

            # Statistics
            total_weights += w.size
            pruned_weights += (w.size - np.count_nonzero(w_pruned))

            new_weights.append(w_pruned)

        # Set the new weights back to the layer
        layer.set_weights(new_weights)

# Save result
model.save(OUTPUT_MODEL)

sparsity = (pruned_weights / total_weights) * 100
print(f"\nPruning Complete.")
print(f"Total Weights: {total_weights}")
print(f"Zeroed Weights: {pruned_weights}")
print(f"Sparsity: {sparsity:.2f}%")
print(f"Saved to: {OUTPUT_MODEL}")