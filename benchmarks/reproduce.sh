#!/bin/bash
# NeuroCUDA — Full reproduction pipeline
# Regenerates every number in the README.
# Requires: Python 3.10+, CUDA GPU recommended (CPU works, slower)
#
# Usage: bash benchmarks/reproduce.sh

set -e
echo "========================================"
echo "NeuroCUDA — Full Reproduction Pipeline"
echo "========================================"

# Ensure data exists
[ -d data/cifar-10-batches-py ] || python3 -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); datasets.CIFAR10('./data', train=False, download=True)"

# ---- Gate 3: QCFS Conversion (3 seeds) ----
echo ""
echo "=== GATE 3: QCFS Conversion Training ==="
for seed in 0 1 2; do
    echo "--- Seed $seed ---"
    python3 gate3_qcfs_convert.py --seed $seed --epochs 30
done

# ---- NIR Round-Trip Verification ----
echo ""
echo "=== NIR: Round-Trip Verification ==="
for seed in 0 1 2; do
    echo "--- Seed $seed ---"
    python3 verify_nir_trained.py --seed $seed
done

# ---- Gate 5: NeuroBench Report ----
echo ""
echo "=== GATE 5: NeuroBench Algorithm Track ==="
python3 gate5_neurobench.py --seeds 0 1 2 --T 32

echo ""
echo "========================================"
echo "Reproduction complete. Results in results/"
echo "========================================"
