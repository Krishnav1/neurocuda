# GATE 3 — PASS

**Date:** 2026-06-19
**Method:** QCFS converter (Bu et al., ICLR 2022)
**Base model:** ResNet-18, 95.56% ± 0.11% ANN (GATE 2)

## Results (3 seeds, full 10K CIFAR-10)

| Seed | T=4 Gap | T=8 Gap | T=16 Gap | T=32 Gap |
|------|---------|---------|----------|----------|
| 0 | 17.20% | 7.21% | 2.72% | 1.11% |
| 1 | 17.16% | 6.07% | 1.95% | **0.90%** |
| 2 | 17.62% | 7.21% | 2.44% | 0.84% |
| **Mean** | **17.33% ± 0.26%** | **6.83% ± 0.66%** | **2.37% ± 0.39%** | **0.95% ± 0.14%** |

## Threshold Verification

17 thresholds per seed, ALL moved on ALL 3 seeds. Frozen-λ bug fixed via separate higher LR on λ parameters.

## Acceptance

Target: gap ≤ 5%. Result: 2.37% at T=16, 0.95% at T=32. **PASS**.
