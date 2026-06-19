# GATE 1 — GROUND TRUTH

**Date:** 2026-06-19
**Test set:** CIFAR-10 full 10,000 images
**Seeds:** 1 (multi-seed deferred to GATE 5)
**Device:** NVIDIA RTX 5050 Laptop GPU

## Current Real Numbers

| Model | Architecture | Params | T | ANN | SNN Conv | SNN FT | Gap |
|-------|-------------|--------|---|-----|----------|--------|-----|
| StrongCNN | 96→192→384 stride-2 | 3.7M | 64 | 80.4% | 62.2% | **74.3%** | **6.1%** |

## Notes

- Single seed only. Multi-seed (≥3) deferred to GATE 5.
- SNN fine-tuned with 3 epochs surrogate gradients.
- Calibration: 95th percentile per layer.
- All numbers measured on full 10K CIFAR-10 test set.

## Known Issues

- **StrideCNN (64→128→256) model file was overwritten** by a weaker training run. Current file gives 67.4% ANN (not 74% as previously reported). The 74% number was from the original better model that was lost.
- **ResNet-style model** cannot be loaded without custom architecture class.
- **MLP/MNIST** not re-measured yet (deferred — MNIST is low priority for GATE 1).

## Next Step

**GATE 2:** Train standard ResNet-18 on CIFAR-10 to ≥93% ANN accuracy. This is a solved problem requiring ~200 epochs with standard recipe.
