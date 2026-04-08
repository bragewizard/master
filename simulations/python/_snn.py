import torch


class SimpleSNN:
    def __init__(self, cnn_model, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.time_limit = 64  # Total ticks per saccade
        self.dt = 1  # Time step
        self.num_hidden = 128
        self.num_output = 10

        # The Window Integrator Thresholds
        # Since we scale weights by 64, these need to be high enough
        # to require "coincident" spikes.
        self.threshold_h = 200.0
        self.threshold_o = 100.0
        # Initialize weights
        self.update_weights(cnn_model)

    def copy_weights_from_fcn(self, fcn):
        """
        Pull weights from the FCN, scale them to a pseudo-i8 range,
        and move them to the GPU.
        """
        with torch.no_grad():
            # Scaling by 64 maps a 2.0 weight to 128.
            self.w1 = (
                (fcn.fc1.weight.data * 64).round().clamp(-128, 127).to(self.device)
            )
            self.w2 = (
                (fcn.fc2.weight.data * 64).round().clamp(-128, 127).to(self.device)
            )

    def encode_to_delays(self, image_tensor):
        """
        Translates intensity (0.0 - 1.0) into temporal delays.
        1.0 (Bright) -> t=0 (Early spike)
        0.0 (Dark)   -> t=32 (Late spike/Silence)
        """
        # Ensure image is a flat 784 tensor on the correct device
        flat_img = image_tensor.view(-1).to(self.device)

        # Linear Delay Encoding
        # We cap the delays at 32 so the integration has time to finish before 64
        delays = ((1.0 - flat_img) * 32).long()
        return delays

    def run_saccade(self, image_tensor):
        """
        Simulates one saccade: Reset -> Integrate -> Spike.
        Every neuron is independent and processed in parallel via GPU.
        """
        # 1. Reset Phase (Phase Ambiguity Fixed by Saccade Start)
        input_delays = self.encode_to_delays(image_tensor)

        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)

        # Track when each neuron fires (-1 = has not fired)
        spikes_h = torch.full(
            (self.num_hidden,), -1, device=self.device, dtype=torch.long
        )
        spikes_o = torch.full(
            (self.num_output,), -1, device=self.device, dtype=torch.long
        )

        # 2. Temporal Loop (The "Clock")
        for t in range(self.time_limit):
            # --- HIDDEN LAYER INTEGRATION ---
            # Find input pixels spiking at THIS tick
            input_mask = (input_delays == t).float()

            if input_mask.any():
                # Parallel Synaptic Summation (Matrix-Vector Multiply)
                v_hidden += torch.mv(self.w1, input_mask)

            # Check for Hidden Spikes (Window Integrator Logic)
            # A neuron fires if potential > threshold AND it hasn't fired yet this saccade
            fired_h = (v_hidden >= self.threshold_h) & (spikes_h == -1)
            spikes_h[fired_h] = t

            # --- OUTPUT LAYER INTEGRATION ---
            # Hidden neurons that just fired now send their "spikes" to the output
            hidden_mask = (spikes_h == t).float()

            if hidden_mask.any():
                v_output += torch.mv(self.w2, hidden_mask)

            # Check for Output Spikes
            fired_o = (v_output >= self.threshold_o) & (spikes_o == -1)
            spikes_o[fired_o] = t

        # Returns the tick time for each of the 10 output neurons
        return spikes_o
