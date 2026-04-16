import torch


class SimpleSNN:
    def __init__(self, cnn_model, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- Network Hyperparameters ---
        self.time_limit = 64  # Total ticks per saccade
        self.dt = 1  # Time step
        self.num_hidden = 128
        self.num_output = 10

        self.threshold_h = {
            "A": 200.0,  # Simple Integrator
            "B": 200.0,  # LIF
            "C": 200.0,  # Linear Ramp
            "D": 200.0,  # Threshold-Sensitive
        }
        self.threshold_o = 100.0
        # --- Neuron Model Parameters ---
        self.tau_m = 20.0  # Leak time constant (Model B)
        self.gamma = 2.0  # State-dependent discount factor (Model D)

        # --- STDP Parameters (Phase III) ---
        self.A_plus = 2.0
        self.A_minus = 1.5
        self.tau_stdp = 10.0
        self.W_max = 128.0

        # Initialize weights
        self.update_weights(cnn_model)

    def update_weights(self, fcn):
        """
        Pull weights from the FCN, scale them to a pseudo-i8 range,
        and move them to the GPU.
        """
        with torch.no_grad():
            self.w1 = (
                (fcn.fc1.weight.data * 64).round().clamp(-128, 127).to(self.device)
            )
            self.w2 = (
                (fcn.fc2.weight.data * 64).round().clamp(-128, 127).to(self.device)
            )

    def encode_to_delays(self, image_tensor):
        """
        Translates intensity (0.0 - 1.0) into temporal delays (TTFS).
        """
        flat_img = image_tensor.view(-1).to(self.device)
        delays = ((1.0 - flat_img) * 32).long()
        return delays

    # =====================================================================
    # NEURON MODELS (Phase I Benchmarks)
    # =====================================================================

    def run_model_a_integrator(self, input_delays, spikes_h, spikes_o):
        """Model A: Simple Window Integrator (Thesis Algorithm 1)"""
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)

        for t in range(self.time_limit):
            # Hidden Layer
            input_mask = (input_delays == t).float()
            if input_mask.any():
                v_hidden += torch.mv(self.w1, input_mask)

            fired_h = (v_hidden >= self.threshold_h["A"]) & (spikes_h == -1)
            spikes_h[fired_h] = t

            # Output Layer
            hidden_mask = (spikes_h == t).float()
            if hidden_mask.any():
                v_output += torch.mv(self.w2, hidden_mask)

            fired_o = (v_output >= self.threshold_o) & (spikes_o == -1)
            spikes_o[fired_o] = t

            if fired_o.any():
                break  # WTA Early Exit

        return spikes_o

    def run_model_b_lif(self, input_delays, spikes_h, spikes_o):
        """Model B: Leaky Integrate-and-Fire (Thesis Algorithm 2)"""
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)
        decay_factor = torch.exp(torch.tensor(-1.0 / self.tau_m, device=self.device))

        for t in range(self.time_limit):
            v_hidden = v_hidden * decay_factor
            v_output = v_output * decay_factor

            # Hidden Layer
            input_mask = (input_delays == t).float()
            if input_mask.any():
                v_hidden += torch.mv(self.w1, input_mask)

            fired_h = (v_hidden >= self.threshold_h["B"]) & (spikes_h == -1)
            spikes_h[fired_h] = t

            # Output Layer
            hidden_mask = (spikes_h == t).float()
            if hidden_mask.any():
                v_output += torch.mv(self.w2, hidden_mask)

            fired_o = (v_output >= self.threshold_o) & (spikes_o == -1)
            spikes_o[fired_o] = t

            if fired_o.any():
                break

        return spikes_o

    def run_model_c_linear_ramp(self, input_delays, spikes_h, spikes_o):
        """Model C: Current-Accumulating Linear Ramp (Thesis Algorithm 3)"""
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        i_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)
        i_output = torch.zeros(self.num_output, device=self.device)

        for t in range(self.time_limit):
            # Apply time-dependent ramp (I * dt, where dt=1)
            v_hidden += i_hidden
            v_output += i_output

            # Hidden Layer
            input_mask = (input_delays == t).float()
            if input_mask.any():
                w_in = torch.mv(self.w1, input_mask)
                v_hidden += w_in
                i_hidden += w_in

            fired_h = (v_hidden >= self.threshold_h["C"]) & (spikes_h == -1)
            spikes_h[fired_h] = t

            # Output Layer
            hidden_mask = (spikes_h == t).float()
            if hidden_mask.any():
                w_hid = torch.mv(self.w2, hidden_mask)
                v_output += w_hid
                i_output += w_hid

            fired_o = (v_output >= self.threshold_o) & (spikes_o == -1)
            spikes_o[fired_o] = t

            if fired_o.any():
                break

        return spikes_o

    def run_model_d_threshold_sensitive(self, input_delays, spikes_h, spikes_o):
        """Model D: Threshold-Sensitive Integration (Thesis Algorithm 4)"""
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)

        for t in range(self.time_limit):
            # Hidden Layer
            input_mask = (input_delays == t).float()
            if input_mask.any():
                discount_h = torch.exp(-self.gamma * (v_hidden / self.threshold_h["D"]))
                w_in = torch.mv(self.w1, input_mask)
                v_hidden += w_in * discount_h

            fired_h = (v_hidden >= self.threshold_h["D"]) & (spikes_h == -1)
            spikes_h[fired_h] = t

            # Output Layer
            hidden_mask = (spikes_h == t).float()
            if hidden_mask.any():
                discount_o = torch.exp(-self.gamma * (v_output / self.threshold_o))
                w_hid = torch.mv(self.w2, hidden_mask)
                v_output += w_hid * discount_o

            fired_o = (v_output >= self.threshold_o) & (spikes_o == -1)
            spikes_o[fired_o] = t

            if fired_o.any():
                break

        return spikes_o

    # =====================================================================
    # SIMULATION ROUTER & PLASTICITY
    # =====================================================================

    def run_saccade(self, image_tensor, model_type="A"):
        """
        Router method for the saccade.
        model_type options: 'A' (Integrator), 'B' (LIF), 'C' (Ramp), 'D' (Threshold)
        """
        input_delays = self.encode_to_delays(image_tensor)
        spikes_h = torch.full(
            (self.num_hidden,), -1, device=self.device, dtype=torch.long
        )
        spikes_o = torch.full(
            (self.num_output,), -1, device=self.device, dtype=torch.long
        )

        if model_type == "A":
            return self.run_model_a_integrator(input_delays, spikes_h, spikes_o)
        elif model_type == "B":
            return self.run_model_b_lif(input_delays, spikes_h, spikes_o)
        elif model_type == "C":
            return self.run_model_c_linear_ramp(input_delays, spikes_h, spikes_o)
        elif model_type == "D":
            return self.run_model_d_threshold_sensitive(
                input_delays, spikes_h, spikes_o
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def apply_stdp(self, pre_spikes, post_spikes, weights):
        """
        TTFS STDP Weight Update (Thesis Algorithm 8)
        """
        with torch.no_grad():
            for i in range(len(pre_spikes)):
                for j in range(len(post_spikes)):
                    t_pre = pre_spikes[i].item()
                    t_post = post_spikes[j].item()

                    # Post-synaptic didn't fire (Penalty)
                    if t_post == -1:
                        weights[j, i] = max(0.0, weights[j, i] - (self.A_minus * 0.1))
                        continue

                    # Pre-synaptic didn't fire (Ignore)
                    if t_pre == -1:
                        continue

                    delta_t = t_post - t_pre

                    # LTP (Pre before Post)
                    if delta_t >= 0:
                        weights[j, i] += self.A_plus * torch.exp(
                            torch.tensor(-delta_t / self.tau_stdp)
                        )
                    # LTD (Post before Pre)
                    else:
                        weights[j, i] -= self.A_minus * torch.exp(
                            torch.tensor(delta_t / self.tau_stdp)
                        )

                    # Clamp to physical range
                    weights[j, i] = max(0.0, min(weights[j, i].item(), self.W_max))
