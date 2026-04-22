import torch


class SimpleSNN:
    def __init__(self, fcn_model, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- Network Structure ---
        self.time_limit = 64
        self.num_hidden = 128
        self.num_output = 10
        self.adaptive_thresholds = torch.zeros(self.num_hidden, device=self.device)

        # --- Model Hyperparameters ---
        self.thresholds = {"A": 200.0, "B": 800.0, "C": 600.0, "D": 200.0}
        self.tau_m = 15.0
        self.coincidence_window = 10

        # --- STDP Params ---
        self.A_plus = 4.0
        self.A_minus = 0.8
        self.W_max = 64.0
        self.W_min = -64.0
        self.theta_plus = 600.0
        self.theta_decay = 0.90

        # -200.0 is a strong inhibitory blast to suppress competitors
        self.w_lat = torch.full(
            (self.num_hidden, self.num_hidden), -200.0, device=self.device
        )
        self.w_lat.fill_diagonal_(0.0)  # Neurons cannot inhibit themselves

        self.ops_count = 0
        self.update_weights(fcn_model)

    def update_weights(self, fcn):
        with torch.no_grad():
            self.w1 = (
                (fcn.fc1.weight.data * 64).round().clamp(-128, 127).to(self.device)
            )
            self.w2 = (
                (fcn.fc2.weight.data * 64).round().clamp(-128, 127).to(self.device)
            )

    def encode_to_delays(self, image_tensor):
        flat_img = image_tensor.view(-1).to(self.device)
        delays = ((1.0 - flat_img) * 32).long()
        delays[flat_img < 0.1] = -1
        return delays

    def apply_stdp(
        self, pre_spikes, post_spikes, weights, k_target=None, use_wta=False
    ):
        with torch.no_grad():
            valid_post = post_spikes.clone()

            if use_wta:
                # OLD: Hard-WTA (Forces exactly 1 winner)
                valid_post[valid_post == -1] = 999
                min_time = valid_post.min()
                if min_time == 999:
                    return []
                winners = torch.where(valid_post == min_time)[0]
                winner_idxs = [winners[torch.randint(len(winners), (1,))].item()]
            else:
                # NEW: Soft-WTA / Lateral Inhibition
                # Anyone who fired gets to learn
                winner_idxs = torch.where(valid_post != -1)[0].tolist()
                if not winner_idxs:
                    return []

            for w_idx in winner_idxs:
                t_fire = post_spikes[w_idx].item()
                ltp_mask = (pre_spikes <= t_fire) & (pre_spikes != -1)
                ltd_mask = (pre_spikes > t_fire) | (pre_spikes == -1)

                weights[w_idx, ltp_mask] += self.A_plus
                weights[w_idx, ltd_mask] -= self.A_minus

                weights[w_idx].clamp_(self.W_min, self.W_max)

                if k_target:
                    current_sum = weights[w_idx].abs().sum()
                    if current_sum > 0:
                        weights[w_idx] *= k_target / current_sum

            return winner_idxs

    def apply_sadp(
        self, pre_spikes, post_spikes, weights, k_target=None, use_wta=False
    ):
        """SADP Approximation for TTFS with Toggleable WTA."""
        with torch.no_grad():
            valid_post = post_spikes.clone()

            if use_wta:
                # --- HARD-WTA: Force exactly 1 winner ---
                valid_post[valid_post == -1] = 999
                min_time = valid_post.min()
                if min_time == 999:
                    return []
                winners = torch.where(valid_post == min_time)[0]
                winner_idxs = [winners[torch.randint(len(winners), (1,))].item()]
            else:
                # --- SOFT-WTA / LATERAL INHIBITION ---
                # Any neuron that survived lateral inhibition and spiked gets to learn
                winner_idxs = torch.where(valid_post != -1)[0].tolist()
                if not winner_idxs:
                    return []

            # SADP Hyperparameter (Agreement window width)
            tau_c = 5.0

            for w_idx in winner_idxs:
                t_fire = post_spikes[w_idx].float()
                pre_f = pre_spikes.float()

                # Calculate absolute time difference (Agreement)
                delta_t = torch.abs(pre_f - t_fire)

                # SADP Math: Correlation decays exponentially with time difference
                ltp_updates = self.A_plus * torch.exp(-delta_t / tau_c)

                valid_pre_mask = pre_spikes != -1

                # 1. Reward agreement (LTP)
                weights[w_idx, valid_pre_mask] += ltp_updates[valid_pre_mask]

                # 2. Punish complete disagreement / silence (LTD)
                weights[w_idx, ~valid_pre_mask] -= self.A_minus

                # Clamp between W_min and W_max
                weights[w_idx].clamp_(self.W_min, self.W_max)

                # Safe Normalization (L1 Norm)
                if k_target:
                    current_sum = weights[w_idx].abs().sum()
                    if current_sum > 0:
                        weights[w_idx] *= k_target / current_sum

            return winner_idxs

    # =====================================================================
    # NEURON MODELS
    # =====================================================================

    def run_model_a_integrator(self, input_delays, spikes_h, spikes_o):
        """Model A: Simple Window Integrator (Infinite Memory)"""
        base_thresh = self.thresholds["A"]
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)

        for t in range(self.time_limit):
            # 1. Hidden Layer
            input_mask = input_delays == t
            num_in_spikes = input_mask.sum().item()
            if num_in_spikes > 0:
                v_hidden += torch.mv(self.w1, input_mask.float())
                self.ops_count += num_in_spikes * self.num_hidden

            eff_thresh_h = base_thresh + self.adaptive_thresholds
            fired_h = (v_hidden >= eff_thresh_h) & (spikes_h == -1)
            spikes_h[fired_h] = t
            self.ops_count += self.num_hidden

            # 2. Output Layer
            hidden_mask = spikes_h == t
            num_hid_spikes = hidden_mask.sum().item()
            if num_hid_spikes > 0:
                v_output += torch.mv(self.w2, hidden_mask.float())
                self.ops_count += num_hid_spikes * self.num_output

            fired_o = (v_output >= base_thresh) & (spikes_o == -1)
            spikes_o[fired_o] = t
            self.ops_count += self.num_output

            if fired_o.any():
                break

        return spikes_o

    def run_model_b_lif(self, input_delays, spikes_h, spikes_o):
        """Model B: Leaky Integrate-and-Fire"""
        base_thresh = self.thresholds["B"]
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)
        decay_factor = torch.exp(torch.tensor(-1.0 / self.tau_m, device=self.device))

        for t in range(self.time_limit):
            # Apply Leak
            v_hidden *= decay_factor
            v_output *= decay_factor
            self.ops_count += self.num_hidden + self.num_output

            # 1. Hidden Layer
            input_mask = input_delays == t
            num_in_spikes = input_mask.sum().item()
            if num_in_spikes > 0:
                v_hidden += torch.mv(self.w1, input_mask.float())
                self.ops_count += num_in_spikes * self.num_hidden

            eff_thresh_h = base_thresh + self.adaptive_thresholds
            fired_h = (v_hidden >= eff_thresh_h) & (spikes_h == -1)
            spikes_h[fired_h] = t
            self.ops_count += self.num_hidden

            # 2. Output Layer
            hidden_mask = spikes_h == t
            num_hid_spikes = hidden_mask.sum().item()
            if num_hid_spikes > 0:
                v_output += torch.mv(self.w2, hidden_mask.float())
                self.ops_count += num_hid_spikes * self.num_output

            fired_o = (v_output >= base_thresh) & (spikes_o == -1)
            spikes_o[fired_o] = t
            self.ops_count += self.num_output

            if fired_o.any():
                break

        return spikes_o

    def run_model_c_linear_ramp(
        self, input_delays, spikes_h, spikes_o, use_lateral=False
    ):
        """Model C: Linear Ramp with Strict Hard Reset Timer"""
        base_thresh = self.thresholds["C"]
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        i_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)
        i_output = torch.zeros(self.num_output, device=self.device)
        timers_h = torch.zeros(self.num_hidden, device=self.device)

        for t in range(self.time_limit):
            v_hidden += i_hidden
            v_output += i_output
            self.ops_count += self.num_hidden + self.num_output

            # Only run if flag is True (Phase 3)
            if use_lateral and t > 0:
                just_fired_h = spikes_h == t - 1
                if just_fired_h.any():
                    lateral_blast = torch.mv(self.w_lat, just_fired_h.float())
                    v_hidden += lateral_blast
                    i_hidden += lateral_blast  # Suppress the slope as well!
            # -------------------------------

            # 1. Hidden Layer & Timers
            input_mask = input_delays == t
            num_in_spikes = input_mask.sum().item()

            if num_in_spikes > 0:
                w_in = torch.mv(self.w1, input_mask.float())
                v_hidden += w_in
                i_hidden += w_in
                self.ops_count += num_in_spikes * self.num_hidden

                start_mask = (w_in != 0) & (timers_h == 0)
                timers_h[start_mask] = self.coincidence_window

            active_mask = timers_h > 0
            num_active_timers = active_mask.sum().item()
            timers_h[active_mask] -= 1
            self.ops_count += num_active_timers

            expired_mask = active_mask & (timers_h == 0)
            v_hidden[expired_mask] = 0.0
            i_hidden[expired_mask] = 0.0

            eff_thresh_h = base_thresh + self.adaptive_thresholds
            fired_h = (v_hidden >= eff_thresh_h) & (spikes_h == -1)
            spikes_h[fired_h] = t
            self.ops_count += self.num_hidden

            # 2. Output Layer
            hidden_mask = spikes_h == t
            num_hid_spikes = hidden_mask.sum().item()

            if num_hid_spikes > 0:
                w_hid = torch.mv(self.w2, hidden_mask.float())
                v_output += w_hid
                i_output += w_hid
                self.ops_count += num_hid_spikes * self.num_output

            fired_o = (v_output >= base_thresh) & (spikes_o == -1)
            spikes_o[fired_o] = t
            self.ops_count += self.num_output

            if fired_o.any():
                break

        return spikes_o

    def run_model_d_threshold_sensitive(self, input_delays, spikes_h, spikes_o):
        """Model D: Threshold-Sensitive Integration"""
        base_thresh = self.thresholds["D"]
        v_hidden = torch.zeros(self.num_hidden, device=self.device)
        v_output = torch.zeros(self.num_output, device=self.device)

        for t in range(self.time_limit):
            # 1. Hidden Layer
            input_mask = input_delays == t
            num_in_spikes = input_mask.sum().item()
            if num_in_spikes > 0:
                eff_thresh_h = base_thresh + self.adaptive_thresholds
                discount_h = torch.exp(-self.gamma * (v_hidden / eff_thresh_h))
                w_in = torch.mv(self.w1, input_mask.float())
                v_hidden += w_in * discount_h
                self.ops_count += num_in_spikes * self.num_hidden + self.num_hidden

            eff_thresh_h = base_thresh + self.adaptive_thresholds
            fired_h = (v_hidden >= eff_thresh_h) & (spikes_h == -1)
            spikes_h[fired_h] = t
            self.ops_count += self.num_hidden

            # 2. Output Layer
            hidden_mask = spikes_h == t
            num_hid_spikes = hidden_mask.sum().item()
            if num_hid_spikes > 0:
                discount_o = torch.exp(-self.gamma * (v_output / base_thresh))
                w_hid = torch.mv(self.w2, hidden_mask.float())
                v_output += w_hid * discount_o
                self.ops_count += num_hid_spikes * self.num_output + self.num_output

            fired_o = (v_output >= base_thresh) & (spikes_o == -1)
            spikes_o[fired_o] = t
            self.ops_count += self.num_output

            if fired_o.any():
                break

        return spikes_o

    def run_saccade(self, image_tensor, model_type="C", use_lateral=False):
        self.ops_count = 0
        input_delays = self.encode_to_delays(image_tensor)
        spikes_h = torch.full(
            (self.num_hidden,), -1, device=self.device, dtype=torch.long
        )
        spikes_o = torch.full(
            (self.num_output,), -1, device=self.device, dtype=torch.long
        )

        if model_type == "A":
            self.run_model_a_integrator(input_delays, spikes_h, spikes_o)
        elif model_type == "B":
            self.run_model_b_lif(input_delays, spikes_h, spikes_o)
        elif model_type == "C":
            self.run_model_c_linear_ramp(
                input_delays, spikes_h, spikes_o, use_lateral=use_lateral
            )
        elif model_type == "D":
            self.run_model_d_threshold_sensitive(input_delays, spikes_h, spikes_o)
        else:
            raise ValueError("Unknown Model Type")

        self.last_spikes_h = spikes_h.clone()
        return spikes_o, self.ops_count

    # =====================================================================
    # TRAINING & EVALUATION
    # =====================================================================

    def evaluate_dataset(self, images, labels, model_type="C"):
        """Standardized evaluation with strict DNF splitting."""
        num_samples = len(labels)
        correct, output_dnfs, hidden_dnfs = 0, 0, 0
        total_ops, spike_times, hidden_spikes_count = [], [], []

        # Rows = Actual Classes, Columns = Predicted Classes
        confusion_matrix = torch.zeros((10, 10), device="cpu")

        ANN_MAC_EQUIV = 101632 * 2

        self.adaptive_thresholds.zero_()

        for i in range(num_samples):
            img_tensor = images[i : i + 1]
            out_spikes, ops = self.run_saccade(img_tensor, model_type=model_type)

            total_ops.append(ops)

            h_spikes = (self.last_spikes_h != -1).sum().item()
            hidden_spikes_count.append(h_spikes)

            if h_spikes == 0:
                hidden_dnfs += 1

            valid_out = out_spikes.clone()
            valid_out[valid_out == -1] = 999
            min_time_out = valid_out.min()

            if min_time_out == 999:
                output_dnfs += 1
            else:
                spike_times.append(min_time_out.item())
                tied_outputs = torch.where(valid_out == min_time_out)[0]
                prediction = tied_outputs[torch.randint(len(tied_outputs), (1,))].item()

                actual = labels[i]
                confusion_matrix[actual, prediction] += 1

                if prediction == labels[i]:
                    correct += 1

        acc = (correct / num_samples) * 100
        mean_ops = sum(total_ops) / len(total_ops)
        eff_gain = ((ANN_MAC_EQUIV - mean_ops) / ANN_MAC_EQUIV) * 100
        mean_latency = sum(spike_times) / len(spike_times) if spike_times else 64
        mean_active_h = sum(hidden_spikes_count) / len(hidden_spikes_count)

        return {
            "Accuracy": acc,
            "Hidden_DNFs": hidden_dnfs,
            "Output_DNFs": output_dnfs,
            "Mean_Latency": mean_latency,
            "Mean_Ops": mean_ops,
            "Efficiency_Gain": eff_gain,
            "Mean_Active_Hidden": mean_active_h,
            "Confusion_Matrix": confusion_matrix.numpy(),
        }
