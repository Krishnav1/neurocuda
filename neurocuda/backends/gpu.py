"""GPU Backend — snnTorch with CUDA acceleration. Production-ready."""
import torch
from typing import Any


class GPUBackend:
    """Runs SNN on GPU using snnTorch. This is the default backend."""

    name = "gpu"
    description = "snnTorch GPU-accelerated SNN simulator (CUDA)"
    is_simulator = True

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compile(self, snn_model) -> Any:
        """Move model to GPU, return compiled model."""
        snn_model.to(self.device)
        snn_model.eval()
        return snn_model

    def run(self, compiled_model, input_data, T: int = 64):
        """Run SNN inference on GPU for T timesteps."""
        with torch.no_grad():
            out = compiled_model(input_data.to(self.device))
        return out

    def benchmark(self, compiled_model, input_shape=(1, 3, 32, 32), T=64, iterations=100):
        """Measure inference latency and throughput."""
        import time
        dummy = torch.randn(*input_shape, device=self.device)
        # Warmup
        for _ in range(10):
            _ = compiled_model(dummy)
        # Benchmark
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = compiled_model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {
            "device": str(self.device),
            "latency_ms": (elapsed / iterations) * 1000,
            "throughput_ips": iterations / elapsed,
            "iterations": iterations,
            "T": T,
        }
