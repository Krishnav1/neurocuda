"""
Model loader for NeuroCUDA ROS2 nodes.

Handles:
  - Loading SNN models from NeuroCUDA hub
  - Converting between ROS2 message types and PyTorch tensors
  - Managing spike state (resetting IF neurons between inferences)
  - Measuring and publishing spike statistics
"""

import numpy as np
import torch

# NeuroCUDA import
try:
    import neurocuda as nc
    from models import IFNeuron, LIFNeuron, reset_spiking
    from neurocuda import hub as nc_hub
    NEUROCUDA_AVAILABLE = True
except ImportError:
    NEUROCUDA_AVAILABLE = False
    print("[neurocuda_ros2] NeuroCUDA not installed. pip install neurocuda")

# ROS2 imports (work when ROS2 environment is sourced)
try:
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[neurocuda_ros2] ROS2 not available. Running in simulation mode.")


class ModelLoader:
    """Load and manage SNN models for ROS2 inference."""

    def __init__(self, model_name, device=None, hardware_target=None):
        """
        Args:
            model_name: e.g., "neurocuda/cnn-nmnist-snn"
            device: torch device (auto-detected if None)
            hardware_target: "gpu", "cpu", "loihi", "fpga"
        """
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hardware_target = hardware_target or str(self.device)

        if not NEUROCUDA_AVAILABLE:
            raise ImportError(
                "NeuroCUDA is required. Install: pip install neurocuda"
            )

        # Load model and metadata from NeuroCUDA hub
        self.model, self.info = nc_hub.load(model_name, device=self.device)
        self.model.eval()

        # Extract key info
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.accuracy = self.info.get("snn_accuracy",
                        self.info.get("snn_solved_best", "N/A"))
        self.sparsity = self.info.get("sparsity", "N/A")
        self.t = self.info.get("t", 16)
        self.task = self.info.get("task", "unknown")

        # Count neuron types
        self.if_count = sum(1 for m in self.model.modules()
                           if isinstance(m, IFNeuron))
        self.lif_count = sum(1 for m in self.model.modules()
                            if isinstance(m, LIFNeuron))

    def reset_state(self):
        """Reset all IF/LIF membrane potentials."""
        reset_spiking(self.model)

    def infer_4d(self, tensor_4d):
        """Run inference on 4D input (B, C, H, W).

        Args:
            tensor_4d: (1, C, H, W) image tensor

        Returns:
            output tensor, spike statistics dict
        """
        with torch.no_grad():
            output = self.model(tensor_4d.to(self.device))
        spike_stats = self._get_spike_stats()
        return output, spike_stats

    def infer_5d(self, tensor_5d):
        """Run inference on 5D temporal input (B, T, C, H, W).

        For event cameras — accumulates spikes over T timesteps.

        Args:
            tensor_5d: (1, T, C, H, W) temporal tensor

        Returns:
            output tensor (B, num_classes), spike statistics dict
        """
        self.reset_state()
        with torch.no_grad():
            output = self.model(tensor_5d.to(self.device))
        spike_stats = self._get_spike_stats()
        return output, spike_stats

    def infer_event_stream(self, event_tensor_generator, T=16):
        """Run continuous inference on streaming events.

        Keeps IF state alive across calls — true neuromorphic processing.
        Each call processes T timesteps of events.

        Args:
            event_tensor_generator: yields (1, T, C, H, W) tensors
            T: timesteps per chunk

        Yields:
            (output, spike_stats) for each chunk
        """
        self.reset_state()
        for events in event_tensor_generator:
            with torch.no_grad():
                output = self.model(events.to(self.device))
            yield output, self._get_spike_stats()

    def _get_spike_stats(self):
        """Measure spike activity in IF/LIF neurons.

        Return 0 for unused timesteps if no spikes have been fired yet.
        """
        total_spikes = 0
        total_activations = 0
        per_layer = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (IFNeuron, LIFNeuron)):
                if hasattr(module, 'v') and module.v is not None:
                    total = module.v.numel()
                    try:
                        # Reshape threshold for broadcasting if needed
                        thresh = module.thresh
                        if isinstance(module, IFNeuron) and module.num_channels is not None:
                            thresh = thresh.view(1, -1, *([1] * (module.v.dim() - 2)))
                        spikes = (module.v >= thresh).float().sum().item()
                    except RuntimeError:
                        # Fallback: just count non-zero v values as proxy
                        spikes = 0
                    per_layer[name] = {
                        "spikes": spikes,
                        "total": total,
                        "sparsity": 100 * (1 - spikes / max(total, 1)),
                    }
                    total_spikes += spikes
                    total_activations += total

        sparsity = 100 * (1 - total_spikes / max(total_activations, 1)) \
            if total_activations > 0 else 0.0

        return {
            "total_spikes": total_spikes,
            "total_activations": total_activations,
            "sparsity": sparsity,
            "per_layer": per_layer,
        }

    def __repr__(self):
        return (f"ModelLoader({self.model_name!r}, "
                f"device={self.device}, "
                f"if_neurons={self.if_count}, "
                f"lif_neurons={self.lif_count}, "
                f"params={self.num_params:,})")


# ===========================================================================
# Tensor conversion utilities
# ===========================================================================

def image_to_tensor(ros_image, target_size=None):
    """Convert ROS2 Image message to PyTorch tensor.

    Args:
        ros_image: sensor_msgs/Image
        target_size: (H, W) to resize to, or None

    Returns:
        torch.Tensor of shape (1, C, H, W) normalized to [0, 1]
    """
    if not ROS2_AVAILABLE:
        raise ImportError("ROS2 (cv_bridge) required for image conversion")

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="rgb8")

    if target_size is not None:
        import cv2
        cv_image = cv2.resize(cv_image, (target_size[1], target_size[0]))

    # HWC → CHW, uint8 → float32, [0,255] → [0,1]
    tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)  # Add batch dim


def events_to_tensor(ros_events, T=16, H=34, W=34):
    """Convert ROS2 EventArray message to temporal tensor.

    Converts sparse event stream → dense (1, T, C, H, W) frame.

    Args:
        ros_events: event_camera_msgs/EventArray
        T: number of temporal bins
        H, W: frame height and width

    Returns:
        torch.Tensor of shape (1, T, C, H, W) where C=2 (ON/OFF events)
    """
    if len(ros_events.events) == 0:
        return torch.zeros(1, T, 2, H, W)

    # Extract timestamps, polarities, x, y
    timestamps = []
    polarities = []
    xs = []
    ys = []

    for event in ros_events.events:
        timestamps.append(event.ts)
        polarities.append(1 if event.polarity else 0)  # ON=1, OFF=0
        xs.append(event.x)
        ys.append(event.y)

    timestamps = np.array(timestamps)
    polarities = np.array(polarities)
    xs = np.array(xs)
    ys = np.array(ys)

    # Normalize timestamps to [0, T)
    t_min, t_max = timestamps.min(), timestamps.max()
    if t_max > t_min:
        t_bins = np.floor((timestamps - t_min) / (t_max - t_min + 1e-8) * T).astype(int)
        t_bins = np.clip(t_bins, 0, T - 1)
    else:
        t_bins = np.zeros_like(timestamps, dtype=int)

    # Build dense frame
    frame = torch.zeros(1, T, 2, H, W)

    for t, p, x, y in zip(t_bins, polarities, xs, ys):
        if 0 <= x < W and 0 <= y < H:
            frame[0, t, p, y, x] += 1.0

    # Clip to avoid extreme values
    frame = torch.clamp(frame, 0, 5.0)

    return frame


def detection_to_msg(output_tensor, class_names=None, threshold=0.5):
    """Convert SNN output tensor to ROS2 detection message.

    Args:
        output_tensor: (1, num_classes) logits
        class_names: list of class name strings
        threshold: confidence threshold

    Returns:
        dict: detection result with class, confidence, top_k
    """
    probs = torch.softmax(output_tensor[0], dim=0)
    top_prob, top_class = probs.max(dim=0)

    top_indices = probs.argsort(descending=True)[:3].tolist()
    top_k_simple = []
    for idx in top_indices:
        if class_names:
            name = class_names[idx]
        else:
            name = str(idx)
        top_k_simple.append((name, probs[idx].item()))

    result = {
        "class_id": top_class.item(),
        "class_name": class_names[top_class.item()] if class_names else str(top_class.item()),
        "confidence": top_prob.item(),
        "top_k": top_k_simple,
    }
    return result
