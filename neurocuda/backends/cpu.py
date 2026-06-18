"""CPU Backend — Pure PyTorch inference, no GPU required. Edge deployment."""
import torch
from typing import Any


class CPUBackend:
    """Runs SNN on CPU. Works everywhere. No CUDA needed."""

    name = "cpu"
    description = "Pure PyTorch CPU SNN inference (works on any machine)"
    is_simulator = True

    def __init__(self):
        self.device = torch.device("cpu")

    def compile(self, snn_model) -> Any:
        """Move model to CPU, optimize for inference."""
        snn_model.to(self.device)
        snn_model.eval()
        return snn_model

    def run(self, compiled_model, input_data, T: int = 64):
        """Run SNN inference on CPU."""
        with torch.no_grad():
            out = compiled_model(input_data.to(self.device))
        return out

    def benchmark(self, compiled_model, input_shape=(1, 3, 32, 32), T=64, iterations=50):
        """Measure CPU inference latency."""
        import time
        dummy = torch.randn(*input_shape)
        for _ in range(5):
            _ = compiled_model(dummy)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = compiled_model(dummy)
        elapsed = time.perf_counter() - start
        return {
            "device": "cpu",
            "latency_ms": (elapsed / iterations) * 1000,
            "throughput_ips": iterations / elapsed,
            "iterations": iterations,
            "T": T,
        }
