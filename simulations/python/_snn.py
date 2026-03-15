import numpy as np


class SpikingNet:
    def __init__(self, input_shape=(64, 64), output_dim=169):
        """
        Simple 2-Layer SNN: Input (64x64) -> Output (e.g., 13x13)
        """
        self.time = 0.0
        self.input_w, self.input_h = input_shape
        self.input_size = self.input_w * self.input_h
        self.output_size = output_dim  # Hidden/Output layer

        self.num_neurons = self.input_size + self.output_size
        self.v = np.zeros(self.num_neurons)
        self.synapses = {}  # {pre_idx: [(post_idx, weight), ...]}

        # Neuron Constants
        self.v_thresh = 1.0
        self.v_reset = 0.0
        self.tau = 5.0
        self.input_currents = np.zeros(self.input_size)

        # Basic local-patch connectivity (to prove it works)
        self._initialize_basic_connectivity()

    def _initialize_basic_connectivity(self):
        """Connects input patches to output neurons."""
        stride = 4  # Downsample 64x64 to 16x16 roughly
        for i in range(self.output_size):
            out_x = (i % 13) * stride
            out_y = (i // 13) * stride
            hidden_idx = self.input_size + i

            # Connect a small 4x4 patch from input to this hidden neuron
            for dx in range(4):
                for dy in range(4):
                    ix, iy = out_x + dx, out_y + dy
                    if ix < self.input_w and iy < self.input_h:
                        pre_idx = iy * self.input_w + ix
                        self._add_connection(pre_idx, hidden_idx, 0.5)

    def _add_connection(self, pre, post, w):
        if pre not in self.synapses:
            self.synapses[pre] = []
        self.synapses[pre].append((post, w))

    def set_input_currents(self, pixel_grid):
        # Flattened single-channel input
        flat = pixel_grid.flatten() / 255.0
        self.input_currents = flat * (self.v_thresh + 0.2)

    def advance(self, dt):
        spiked_indices = []
        # 1. Update Input Layer
        self.v[: self.input_size] += (dt / self.tau) * (
            -self.v[: self.input_size] + self.input_currents
        )
        in_spikes = np.where(self.v[: self.input_size] >= self.v_thresh)[0]

        for pre_idx in in_spikes:
            spiked_indices.append(pre_idx)
            self.v[pre_idx] = self.v_reset
            if pre_idx in self.synapses:
                for post_idx, w in self.synapses[pre_idx]:
                    self.v[post_idx] += w

        # 2. Update Output Layer
        out_start = self.input_size
        self.v[out_start:] += (dt / self.tau) * (-self.v[out_start:])
        out_spikes = np.where(self.v[out_start:] >= self.v_thresh)[0]

        for idx in out_spikes:
            real_idx = idx + out_start
            spiked_indices.append(real_idx)
            self.v[real_idx] = self.v_reset

        return spiked_indices, len(in_spikes)
