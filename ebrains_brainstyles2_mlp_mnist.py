"""
EBRAINS BrainScaleS-2: MLP MNIST SNN inference on real analog silicon.
=============================================================
Copy-paste each cell into your EBRAINS Lab notebook (EBRAINS-experimental kernel).

PREREQUISITE: Upload mlp_mnist_weights.npz to your Collab Drive first.
    Collab → Drive (left menu) → Upload → select mlp_mnist_weights.npz

Then in EBRAINS Lab, run these cells in order.
"""

# ============================================================================
# CELL 1: Setup hardware + check weights file
# ============================================================================
from _static.common.helpers import setup_hardware_client
setup_hardware_client()

import pynn_brainscales.brainscales2 as pynn
import numpy as np
import os

# Path to your uploaded weights in Collab Drive
WEIGHTS_PATH = "/mnt/user/shared/Neuromorphic Computing in EBRAINS/mlp_mnist_weights.npz"
# If you uploaded to your own Collab, use:
# WEIGHTS_PATH = "/mnt/user/shared/YOUR_COLLAB_NAME/mlp_mnist_weights.npz"

if not os.path.exists(WEIGHTS_PATH):
    print(f"ERROR: {WEIGHTS_PATH} not found!")
    print("Upload mlp_mnist_weights.npz to your Collab Drive first.")
else:
    data = np.load(WEIGHTS_PATH)
    print("Weights loaded:", list(data.keys()))
    for k in data:
        print(f"  {k}: {data[k].shape}")


# ============================================================================
# CELL 2: Load MNIST test image
# ============================================================================
# Download MNIST directly in Lab (no torch needed)
import gzip
import urllib.request

def load_mnist_image(idx):
    """Load one MNIST test image (pure numpy, no torch)."""
    cache = "/tmp/mnist_test.npz"

    if not os.path.exists(cache):
        print("Downloading MNIST test set (~2.5MB)...")
        # Download gzipped files
        img_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
        lbl_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

        for url, name in [(img_url, "img"), (lbl_url, "lbl")]:
            gz_path = f"/tmp/mnist_{name}.gz"
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, 'rb') as f:
                if name == "img":
                    magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
                    data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows*cols)
                else:
                    magic, num = np.frombuffer(f.read(8), dtype='>i4')
                    data = np.frombuffer(f.read(), dtype=np.uint8)
            np.savez_compressed(cache, images=data, labels=data)
            print(f"Saved {name} to {cache}")

    arr = np.load(cache)
    img = arr["images"][idx].astype(np.float32) / 255.0  # [0,1]
    label = arr["labels"][idx]
    print(f"Image {idx}: label={label}, pixel range=[{img.min():.3f}, {img.max():.3f}]")
    return img, label

# Load first test image (digit 7)
test_img, test_label = load_mnist_image(0)
print(f"Test image shape: {test_img.shape}")


# ============================================================================
# CELL 3: Convert weights to PyNN connection lists
# ============================================================================
def weight_matrix_to_list(w):
    """Convert PyTorch weight matrix to PyNN FromListConnector format.

    PyTorch: w[out, in] — weight from input j to output i
    PyNN FromListConnector: [(target_idx, source_idx, weight, delay), ...]
    """
    out_neurons, in_neurons = w.shape
    conn_list = []
    for i in range(out_neurons):
        for j in range(in_neurons):
            conn_list.append((i, j, float(w[i, j]), 1.0))
    return conn_list

data = np.load(WEIGHTS_PATH)
w1 = data["fc1_weight"]  # 256 x 784
w2 = data["fc2_weight"]  # 256 x 256
w3 = data["fc3_weight"]  # 10 x 256
b1 = data["fc1_bias"]     # 256
b2 = data["fc2_bias"]     # 256
b3 = data["fc3_bias"]     # 10

print(f"FC1: {w1.shape} (256 out × 784 in)")
print(f"FC2: {w2.shape} (256 out × 256 in)")
print(f"FC3: {w3.shape} (10 out × 256 in)")

# For a quick test: use SUBSET of neurons to stay within BrainScaleS-2 limits
# BrainScaleS-2 has ~512 neurons per chip
# We'll use: 64 inputs → 32 hidden → 10 output = 106 neurons (fits on one chip)
N_IN = 64
N_HID = 32
N_OUT = 10

# Take subset of first N_IN pixels + first N_HID hidden neurons
w1_sub = w1[:N_HID, :N_IN]    # 32 x 64
w2_sub = w2[:N_HID, :N_HID]   # 32 x 32 (use same N_HID for both hidden layers — simplified)
w3_sub = w3[:N_OUT, :N_HID]    # 10 x 32

print(f"\nSubset sizes (fits on 1 chip):")
print(f"  FC1: {w1_sub.shape} ({N_HID*N_IN} connections)")
print(f"  FC2: {w2_sub.shape} ({N_HID*N_HID} connections)")
print(f"  FC3: {w3_sub.shape} ({N_OUT*N_HID} connections)")
print(f"  Total neurons: {N_IN + N_HID + N_HID + N_OUT}")
print(f"  Total connections: {N_HID*N_IN + N_HID*N_HID + N_OUT*N_HID}")

# Convert to PyNN connection lists (only for small subset — full would be too slow)
conn1 = weight_matrix_to_list(w1_sub)
conn2 = weight_matrix_to_list(w2_sub)
conn3 = weight_matrix_to_list(w3_sub)
print("Connection lists built ✅")


# ============================================================================
# CELL 4: Build BrainScaleS-2 SNN and run
# ============================================================================
pynn.setup()

# --- Populations ---
# Input: convert MNIST pixel intensities to Poisson rates
pixel_rates = test_img[:N_IN] * 1000.0  # Scale 0-1 to 0-1000 Hz
pop_in = pynn.Population(N_IN, pynn.cells.SpikeSourcePoisson(
    rate=pixel_rates.tolist(),
    start=0.0,
    duration=100.0
))

# Hidden layer 1
pop_hid1 = pynn.Population(N_HID, pynn.cells.HXNeuron(
    leak_v_leak=400,
    threshold_v_threshold=600,
    threshold_enable=True,
    excitatory_input_enable=True,
))

# Hidden layer 2
pop_hid2 = pynn.Population(N_HID, pynn.cells.HXNeuron(
    leak_v_leak=400,
    threshold_v_threshold=600,
    threshold_enable=True,
    excitatory_input_enable=True,
))

# Output layer
pop_out = pynn.Population(N_OUT, pynn.cells.HXNeuron(
    leak_v_leak=400,
    threshold_v_threshold=600,
    threshold_enable=True,
    excitatory_input_enable=True,
))

# --- Projections with trained weights ---
pynn.Projection(pop_in, pop_hid1, pynn.FromListConnector(conn1))
pynn.Projection(pop_hid1, pop_hid2, pynn.FromListConnector(conn2))
pynn.Projection(pop_hid2, pop_out, pynn.FromListConnector(conn3))

# --- Record output spikes ---
pop_out.record("spikes")

# --- Run ---
print(f"\nRunning SNN on BrainScaleS-2 (analog silicon, Heidelberg)...")
print(f"Input: MNIST digit {test_label}, {N_IN} pixels")
print(f"Network: {N_IN} → {N_HID} → {N_HID} → {N_OUT}")
pynn.run(100.0)

# --- Results ---
spikes = pop_out.get_data("spikes")
print("\n=== RESULTS ===")
print(f"Image: MNIST test[{0}], True label: {test_label}")
print(f"BrainScaleS-2 hardware spikes:")

spike_counts = []
for i, train in enumerate(spikes.segments[0].spiketrains):
    count = len(train)
    spike_counts.append(count)
    print(f"  Neuron {i} (digit {i}): {count} spikes")

# Predict: neuron with most spikes wins
prediction = np.argmax(spike_counts)
print(f"\nPrediction: {prediction} (most spikes = {spike_counts[prediction]})")
print(f"True label: {test_label}")
print(f"CORRECT ✅" if prediction == test_label else f"WRONG ❌")

pynn.end()


# ============================================================================
# CELL 5 (OPTIONAL): Run full test set (first 100 images)
# ============================================================================
"""
To run more images, wrap Cell 4 in a loop:

results = []
for img_idx in range(100):
    img, lbl = load_mnist_image(img_idx)
    # ... build network, run, get prediction ...
    results.append((lbl, prediction))

correct = sum(1 for l, p in results if l == p)
print(f"BrainScaleS-2 accuracy (100 images): {correct}%")
"""
