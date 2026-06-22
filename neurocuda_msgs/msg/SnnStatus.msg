# SNN model status — published periodically
string model_name       # e.g. "neurocuda/cnn-nmnist-snn"
string task             # e.g. "event-camera-vision"
string architecture     # e.g. "3-layer CNN with IF neurons"
float32 accuracy        # model accuracy %
int32 total_params      # number of parameters
int32 neuron_count      # total IF/LIF neurons
string device           # "cuda", "cpu"
float32 avg_sparsity    # running average sparsity %
float32 inference_time_ms # last inference time in ms
