import numpy as np
import torch
import heapq

# --- CONFIG ---
# Must match the CNN architecture
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
KERNEL_SIZE = 3
STRIDE = 2
NUM_FILTERS = 4

# Output map size after stride 2 convolution
# (28 - 3 + 2*0) / 2 + 1 = 13.5 -> floor(13)
MAP_WIDTH = 13
MAP_HEIGHT = 13

# Indices ranges
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
HIDDEN_SIZE = NUM_FILTERS * MAP_WIDTH * MAP_HEIGHT
OUTPUT_SIZE = 2  # X, Y

# Neuron Parameters
V_THRESH = 1.0  # Normalized threshold
V_RESET = 0.0
TAU = 5.0  # Membrane time constant
DT = 1.0  # Simulation step size


class SpikingNet:
    def __init__(self):
        self.time = 0.0

        # 1. Neuron State
        # 0 -> INPUT_SIZE-1 : Input Neurons (Retina)
        # INPUT_SIZE -> +HIDDEN_SIZE : Conv Neurons
        # ... -> +OUTPUT_SIZE : Readout Neurons (we won't spike these, just accumulate)
        self.num_neurons = INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE
        self.v = np.zeros(self.num_neurons)

        # 2. Connectivity: Adjacency List
        # { pre_idx: [(post_idx, weight), ...] }
        self.synapses = {}

        # 3. Input Currents (constant feed for Rate Coding)
        self.input_currents = np.zeros(INPUT_SIZE)

        # 4. Metrics
        self.total_synaptic_ops = 0
        self.active_synapses_count = 0  # Static count of connections

    def load_cnn_weights(self, filepath="cnn_weights.pth"):
        print(f"[SNN] Loading weights from {filepath}...")
        try:
            state_dict = torch.load(filepath)
        except FileNotFoundError:
            print("[SNN] Error: Weights file not found! Run cnn_baseline.py first.")
            return

        # --- A. Load CONV Weights ---
        # Shape: [4, 1, 3, 3] -> (Out_Channels, In_Channels, H, W)
        conv_weights = state_dict["conv1.weight"].numpy()

        count = 0
        # Iterate over every hidden neuron (Feature Map)
        for f in range(NUM_FILTERS):
            for my in range(MAP_HEIGHT):
                for mx in range(MAP_WIDTH):
                    # Calculate this neuron's global index
                    hidden_idx = (
                        INPUT_SIZE
                        + (f * MAP_HEIGHT * MAP_WIDTH)
                        + (my * MAP_WIDTH)
                        + mx
                    )

                    # Determine the Top-Left corner of the receptive field in Input
                    # Stride = 2
                    input_y_start = my * STRIDE
                    input_x_start = mx * STRIDE

                    # Connect to the 3x3 patch in Input
                    for ky in range(KERNEL_SIZE):
                        for kx in range(KERNEL_SIZE):
                            iy = input_y_start + ky
                            ix = input_x_start + kx

                            # Boundary check (though logic should keep it inside)
                            if 0 <= iy < INPUT_HEIGHT and 0 <= ix < INPUT_WIDTH:
                                input_idx = (iy * INPUT_WIDTH) + ix
                                w = conv_weights[f, 0, ky, kx]

                                # Add Synapse: Input -> Hidden
                                self._add_connection(input_idx, hidden_idx, w)
                                count += 1

        print(f"[SNN] Wired {count} Conv synapses.")

        # --- B. Load FC Weights (Optional for visual) ---
        # Mapping the final readout is complex in SNNs (usually done by rate decoding).
        # For this visual demo, we focus on the efficiency of the CONV layer (the expensive part).
        self.active_synapses_count = count

    def _add_connection(self, pre, post, w):
        if pre not in self.synapses:
            self.synapses[pre] = []
        self.synapses[pre].append((post, w))

    def set_input_currents(self, pixel_grid):
        """
        pixel_grid: 28x28 numpy array (0-255)
        """
        # Normalize to 0-1 range, then scale to input current
        # A pixel of 255 should guarantee firing.
        flat = pixel_grid.flatten() / 255.0
        self.input_currents = flat * (V_THRESH + 0.5)  # Slight overdrive

    def advance(self, dt):
        """
        Euler integration step for LIF neurons.
        Returns: list of spiked_indices
        """
        spiked_indices = []

        # 1. Update Input Layer (driven by external current)
        # dV = (-V + I) / Tau
        # We only simulate Input neurons here because they drive the rest
        input_range = range(INPUT_SIZE)

        # Vectorized update for Input Layer
        # v[t+1] = v[t] + dt/tau * (-v[t] + I)
        self.v[:INPUT_SIZE] += (dt / TAU) * (-self.v[:INPUT_SIZE] + self.input_currents)

        # Check for spikes in Input Layer
        spikes = np.where(self.v[:INPUT_SIZE] >= V_THRESH)[0]

        current_ops = 0

        # Process Spikes
        for pre_idx in spikes:
            spiked_indices.append(pre_idx)
            self.v[pre_idx] = V_RESET  # Reset

            # Propagate Spikes (Instantaneous for simplicity, or queue them)
            if pre_idx in self.synapses:
                connections = self.synapses[pre_idx]

                # EFFICIENCY METRIC:
                # 1 Input Spike -> N Hidden Neurons
                # This matches the MACs saved vs CNN
                ops = len(connections)
                current_ops += ops

                # Apply weights to post-synaptic neurons
                for post_idx, w in connections:
                    self.v[post_idx] += w  # Integrate Weight immediately

        self.total_synaptic_ops += current_ops

        # 2. Update Hidden Layer (Leaky decay only, input came from spikes above)
        hidden_start = INPUT_SIZE
        hidden_end = INPUT_SIZE + HIDDEN_SIZE
        self.v[hidden_start:hidden_end] += (dt / TAU) * (
            -self.v[hidden_start:hidden_end]
        )

        # Check for spikes in Hidden Layer
        hidden_spikes = np.where(self.v[hidden_start:hidden_end] >= V_THRESH)[0]
        for idx in hidden_spikes:
            real_idx = idx + hidden_start
            spiked_indices.append(real_idx)
            self.v[real_idx] = V_RESET
            # If we had a next layer, we would propagate here

        return spiked_indices, current_ops

    def get_fan_out_count(self, neuron_idx):
        if neuron_idx in self.synapses:
            return len(self.synapses[neuron_idx])
        return 0
