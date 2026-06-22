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

# ROS2 imports
ROS2_AVAILABLE = False
BRIDGE_AVAILABLE = False

try:
    from sensor_msgs.msg import Image
    from neurocuda_msgs.msg import SnnDetection, SnnSpikeEvent, SnnStatus
    ROS2_AVAILABLE = True
except ImportError:
    print("[neurocuda_ros2] ROS2 core not available. Running in simulation mode.")

try:
    from cv_bridge import CvBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    print("[neurocuda_ros2] cv_bridge not available. Using numpy-only image conversion.")


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

    def _get_neuron_type(self, layer_name):
        """Return neuron type string for a named layer."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                if isinstance(module, LIFNeuron):
                    return "LIF"
                if isinstance(module, IFNeuron):
                    return "IF"
        return "unknown"

    def _describe_architecture(self):
        """Return a short architecture description string."""
        parts = []
        for name, module in self.model.named_children():
            parts.append(type(module).__name__)
        return " → ".join(parts) if parts else "unknown"


# ===========================================================================
# Tensor conversion utilities
# ===========================================================================

def image_to_tensor(ros_image, target_size=None):
    """Convert ROS2 Image message to PyTorch tensor.

    Works with OR without cv_bridge — uses direct numpy conversion as fallback.

    Args:
        ros_image: sensor_msgs/Image
        target_size: (H, W) to resize to, or None

    Returns:
        torch.Tensor of shape (1, C, H, W) normalized to [0, 1]
    """
    # Extract raw bytes from ROS Image message
    h = ros_image.height
    w = ros_image.width
    encoding = ros_image.encoding
    data = ros_image.data
    step = ros_image.step  # row stride in bytes

    # Determine channels from encoding
    if encoding == "rgb8":
        channels = 3
        dtype = np.uint8
    elif encoding == "rgba8":
        channels = 4
        dtype = np.uint8
    elif encoding == "mono8" or encoding == "8UC1":
        channels = 1
        dtype = np.uint8
    elif encoding == "bgr8":
        channels = 3
        dtype = np.uint8
    elif encoding == "32FC1":
        channels = 1
        dtype = np.float32
    else:
        # Default: assume rgb8
        channels = 3
        dtype = np.uint8

    # Convert raw bytes to numpy array
    np_arr = np.frombuffer(data, dtype=dtype).reshape(h, step // (step // w), -1)

    # If step != width * channels, trim padding
    expected_cols = w * (np.dtype(dtype).itemsize) * channels // (np.dtype(dtype).itemsize)
    if step != expected_cols:
        # Handle padded rows by taking only the valid columns
        np_arr = np.frombuffer(data, dtype=dtype)
        np_arr = np_arr.reshape(h, w, channels) if channels > 1 else np_arr.reshape(h, w)
    else:
        np_arr = np.frombuffer(data, dtype=dtype).reshape(h, w, channels) if channels > 1 else np.frombuffer(data, dtype=dtype).reshape(h, w)

    # Ensure 3 channels
    if len(np_arr.shape) == 2:
        np_arr = np.stack([np_arr] * 3, axis=-1)
    elif np_arr.shape[-1] == 4:
        np_arr = np_arr[:, :, :3]  # drop alpha

    # Resize if needed (using simple numpy slicing or nearest-neighbor)
    if target_size is not None:
        th, tw = target_size
        # Simple resize: crop center or tile
        if th < h and tw < w:
            y0, x0 = (h - th) // 2, (w - tw) // 2
            np_arr = np_arr[y0:y0 + th, x0:x0 + tw]
        elif th > h or tw > w:
            np_arr = np.pad(np_arr, ((0, max(0, th - h)), (0, max(0, tw - w)), (0, 0)),
                           mode='edge')[:th, :tw]

    # HWC → CHW, uint8 → float32, [0,255] → [0,1]
    tensor = torch.from_numpy(np_arr.copy()).permute(2, 0, 1).float() / 255.0
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
    """Convert SNN output tensor to SnnDetection message or dict.

    Args:
        output_tensor: (1, num_classes) logits
        class_names: list of class name strings
        threshold: confidence threshold

    Returns:
        SnnDetection (if ROS2 available) or dict
    """
    probs = torch.softmax(output_tensor[0], dim=0)
    top_prob, top_class = probs.max(dim=0)

    k = min(3, len(probs))
    top_indices = probs.argsort(descending=True)[:k]

    if ROS2_AVAILABLE:
        msg = SnnDetection()
        msg.class_id = top_class.item()
        msg.class_name = class_names[top_class.item()] if class_names else str(top_class.item())
        msg.confidence = top_prob.item()
        for idx in top_indices:
            name = class_names[idx] if class_names else str(idx)
            msg.top_k_labels.append(name)
            msg.top_k_scores.append(probs[idx].item())
        return msg

    # Fallback for non-ROS2 environment
    top_k_simple = []
    for idx in top_indices:
        name = class_names[idx] if class_names else str(idx)
        top_k_simple.append((name, probs[idx].item()))

    return {
        "class_id": top_class.item(),
        "class_name": class_names[top_class.item()] if class_names else str(top_class.item()),
        "confidence": top_prob.item(),
        "top_k": top_k_simple,
    }


def spike_stats_to_msg(spike_stats, model_loader=None):
    """Convert spike statistics to a list of SnnSpikeEvent messages.

    Args:
        spike_stats: dict from ModelLoader._get_spike_stats()
        model_loader: optional ModelLoader instance for accuracy

    Returns:
        list of SnnSpikeEvent messages
    """
    messages = []
    if ROS2_AVAILABLE:
        for layer_name, data in sorted(spike_stats.get("per_layer", {}).items()):
            msg = SnnSpikeEvent()
            msg.layer_name = layer_name
            msg.neuron_type = model_loader._get_neuron_type(layer_name) if model_loader else "unknown"
            msg.spike_count = int(data["spikes"])
            msg.total_neurons = int(data["total"])
            msg.sparsity = float(data["sparsity"])
            messages.append(msg)
    return messages


def status_to_msg(model_loader, spike_stats, inference_time_ms=0.0):
    """Build an SnnStatus message from model metadata and current spike stats.

    Args:
        model_loader: ModelLoader instance
        spike_stats: dict from _get_spike_stats()
        inference_time_ms: last inference time in milliseconds

    Returns:
        SnnStatus message (if ROS2 available) or dict
    """
    if ROS2_AVAILABLE:
        msg = SnnStatus()
        msg.model_name = model_loader.model_name
        msg.task = model_loader.task
        msg.architecture = model_loader._describe_architecture()
        msg.accuracy = float(model_loader.accuracy) if isinstance(model_loader.accuracy, (int, float)) else 0.0
        msg.total_params = model_loader.num_params
        msg.neuron_count = model_loader.if_count + model_loader.lif_count
        msg.device = str(model_loader.device)
        msg.avg_sparsity = float(spike_stats.get("sparsity", 0.0))
        msg.inference_time_ms = float(inference_time_ms)
        return msg

    return {
        "model_name": model_loader.model_name,
        "task": model_loader.task,
        "accuracy": model_loader.accuracy,
        "total_params": model_loader.num_params,
        "neuron_count": model_loader.if_count + model_loader.lif_count,
        "device": str(model_loader.device),
        "avg_sparsity": spike_stats.get("sparsity", 0.0),
        "inference_time_ms": inference_time_ms,
    }
