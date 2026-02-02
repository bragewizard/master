import numpy as np
from numba import jit, prange

def intensity_to_delay_encoding(image, T_max=100, T_min=0):
    normalized_image = image.astype(float) / 255.0
    spike_times = T_max - (T_max - T_min) * normalized_image
    # print(spike_times.shape)
    # data = spike_times[(spike_times < 100)]
    # print(data.shape)
    return spike_times

def negative_image_encoding(image, T_max=100, T_min=0):
    negative_image = 255 - image
    return intensity_to_delay_encoding(negative_image, T_max=T_max, T_min=T_min)

def integrate_and_fire(excitatory, inhibitory, threshold=1.0):
    excitatory_spikes = [(t, 1) for t in excitatory.flatten() if not np.isnan(t)]
    if inhibitory is None:
        inhibitory_spikes = []
    else:
        inhibitory_spikes = [(t, -1) for t in inhibitory.flatten() if not np.isnan(t)]
    all_spikes = excitatory_spikes + inhibitory_spikes
    if not all_spikes:
        return None
    all_spikes.sort(key=lambda x: x[0])
    integrated_potential = 0.0
    firing_time = None
    for time, spike_type in all_spikes:
        integrated_potential += spike_type
        integrated_potential = max(0, integrated_potential)
        if integrated_potential >= threshold:
            firing_time = time
            break
    return firing_time

def leaky_integrate_and_fire(excitatory_spikes, inhibitory_spikes, weights_exc, weights_inh,
                                T_sim=100, dt=1.0,
                                tau_m=10.0, V_thresh=15.0, V_rest=0.0,
                                tau_syn_exc=5.0, tau_syn_inh=5.0, return_trace=False):
    time_steps = np.arange(0, T_sim + dt, dt)
    voltage = np.full_like(time_steps, V_rest)
    I_exc = np.zeros_like(time_steps)
    I_inh = np.zeros_like(time_steps)
    decay_exc = np.exp(-dt / tau_syn_exc)
    decay_inh = np.exp(-dt / tau_syn_inh)
    spike_dict_exc = {int(t / dt): w for t, w in zip(excitatory_spikes, weights_exc) if not np.isnan(t)}
    spike_dict_inh = {int(t / dt): w for t, w in zip(inhibitory_spikes, weights_inh) if not np.isnan(t)}
    for i in range(1, len(time_steps)):
        I_exc[i] = I_exc[i-1] * decay_exc
        I_inh[i] = I_inh[i-1] * decay_inh
        
        if (i-1) in spike_dict_exc:
            I_exc[i] += spike_dict_exc[i-1]
        if (i-1) in spike_dict_inh:
            I_inh[i] += spike_dict_inh[i-1]
            
        total_current = I_exc[i] + I_inh[i]

        dV = (-(voltage[i-1] - V_rest) + total_current) / tau_m * dt
        voltage[i] = voltage[i-1] + dV
        
        if voltage[i] >= V_thresh:
            if return_trace:
                voltage[i] = 40.0
                voltage[i-1] = V_rest
            else:
                return time_steps[i]

    if return_trace:
        return time_steps, voltage, I_exc, I_inh
    return None

def get_postsynaptic_events(
    input_spike_times: np.ndarray,
    input_spike_indices: np.ndarray,
    connections: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrival_times = []
    target_indices = []
    weights = []
    for i in range(len(input_spike_times)):
        pre_syn_time = input_spike_times[i]
        pre_syn_index = input_spike_indices[i]
        if pre_syn_index in connections:
            for post_syn_index, weight, delay in connections[pre_syn_index]:
                arrival_times.append(pre_syn_time + delay)
                target_indices.append(post_syn_index)
                weights.append(weight)
    return np.array(arrival_times), np.array(target_indices), np.array(weights)

def create_conv_connections(input_shape, output_shape):
    input_h, input_w = input_shape
    output_h, output_w = output_shape
    connections = {}
    for y in range(output_h):
        for x in range(output_w):
            on_neuron_idx = y * output_w + x
            off_neuron_idx = (output_h * output_w) + on_neuron_idx
            for dy in range(3):
                for dx in range(3):
                    pre_syn_y, pre_syn_x = y + dy, x + dx
                    pre_syn_idx = pre_syn_y * input_w + pre_syn_x
                    if pre_syn_idx not in connections:
                        connections[pre_syn_idx] = []
                    if dx == 1 and dy == 1:
                        connections[pre_syn_idx].append((on_neuron_idx, 1.0, 0.0))
                        connections[pre_syn_idx].append((off_neuron_idx, -1.0, 0.0))
                    else:
                        connections[pre_syn_idx].append((on_neuron_idx, -1.0, 0.0))
                        connections[pre_syn_idx].append((off_neuron_idx, 1.0, 0.0))
    return connections
