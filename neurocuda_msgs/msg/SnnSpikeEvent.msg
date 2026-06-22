# Per-layer spike activity
string layer_name      # e.g. "act1", "act2"
string neuron_type     # "IF" or "LIF"
int32 spike_count      # number of spikes in this layer
int32 total_neurons    # total neurons in this layer
float32 sparsity       # (1 - spike_count/total_neurons) * 100
