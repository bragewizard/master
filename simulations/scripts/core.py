import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
from PIL import Image
from visualize import visualize3dSpikes, visualizeRasterPlot, visualizeSequenceRasterPlot,\
visualizeImage, visualizeSpikeTimes, visualizeStaticRasterPlot, visualizeSpikes,\
visualizeStimuliAnimation,plotNeuronVoltageTrace
from data import generate_checkerboard, generate_spatiotemporal_stimuli

def intensityToDelayEncoding(image, T_max=100, T_min=0):
    normalized_image = image.astype(float) / 255.0
    spike_times = T_max - (T_max - T_min) * normalized_image
    return spike_times

def negativeImageEncoding(image, T_max=100, T_min=0):
    negative_image = 255 - image
    return intensityToDelayEncoding(negative_image, T_max=T_max, T_min=T_min)

def integrateAndFire(excitatory, inhibitory, threshold=1.0):
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

def leakyIntegrateAndFire(excitatory_spikes, inhibitory_spikes, weights_exc, weights_inh, 
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

def locationWiseWTA(list_of_spike_time_grids):
    if not list_of_spike_time_grids:
        return np.array([])
    combined_min_times = list_of_spike_time_grids[0]
    for i in range(1, len(list_of_spike_time_grids)):
        combined_min_times = np.fmin(combined_min_times, list_of_spike_time_grids[i])
    return combined_min_times

def visualizeRendomStimuli():
    pattern_to_inject = np.array([
        [10, np.nan, np.nan],
        [np.nan, 20, np.nan],
        [np.nan, np.nan, 30]
    ])
    stimuli = generate_spatiotemporal_stimuli(
        sequence_length=50,
        grid_size=(32, 32),
        pattern=pattern_to_inject,
        num_injections=8,
        background_spike_prob=0.002,
        T_max=100 # This T_max will be our frame_duration
    )
    visualizeSequenceRasterPlot(stimuli, frame_duration=100)

def createOnOffFilters(center_weight=8, surround_weight=-1):
    on_filter = np.full((3, 3), surround_weight, dtype=float)
    on_filter[1, 1] = center_weight
    off_filter = np.full((3, 3), -surround_weight, dtype=float) # Note the sign flip
    off_filter[1, 1] = -center_weight
    return on_filter, off_filter    

def applyReceptiveFields(image, on_filter, off_filter, T_max=100, T_min=0):
    height, width = image.shape
    output_height = height - 2
    output_width = width - 2
    on_spike_times = np.full((output_height, output_width), np.nan)
    off_spike_times = np.full((output_height, output_width), np.nan)
    normalized_image = image.astype(float) / 255.0
    max_potential = on_filter[1, 1]
    for y in range(output_height):
        for x in range(output_width):
            patch = normalized_image[y:y+3, x:x+3]
            on_potential = np.sum(patch * on_filter)
            off_potential = np.sum(patch * off_filter)
            if on_potential > 0:
                scaled_potential = min(on_potential / max_potential, 1.0)
                on_spike_times[y, x] = T_max - (T_max - T_min) * scaled_potential
            if off_potential > 0:
                scaled_potential = min(off_potential / max_potential, 1.0)
                off_spike_times[y, x] = T_max - (T_max - T_min) * scaled_potential
    return on_spike_times, off_spike_times

def applyReceptiveFieldsLIF(input_spikes, V_thresh=15.0):
    height, width = input_spikes.shape
    output_height = height - 2
    output_width = width - 2
    on_spike_times = np.full((output_height, output_width), np.nan)
    off_spike_times = np.full((output_height, output_width), np.nan)
    W_CENTER_EXC = 60.0
    W_SURROUND_INH = -7.5
    on_exc_weights = np.array([W_CENTER_EXC])
    on_inh_weights = np.full(8, W_SURROUND_INH)
    off_exc_weights = np.full(8, -W_SURROUND_INH)
    off_inh_weights = np.array([-W_CENTER_EXC])
    neuron_params = {'tau_syn_exc': 5.0, 'tau_syn_inh': 8.0, 'tau_m': 10.0}
    for y in range(output_height):
        for x in range(output_width):
            patch = input_spikes[y:y+3, x:x+3]
            center_spike = patch[1, 1]
            surround_spikes = np.delete(patch.flatten(), 4)
            on_spike_times[y, x] = leakyIntegrateAndFire(
                excitatory_spikes=np.array([center_spike]), inhibitory_spikes=surround_spikes,
                weights_exc=on_exc_weights, weights_inh=on_inh_weights, 
                V_thresh=V_thresh, **neuron_params)
            off_spike_times[y, x] = leakyIntegrateAndFire(
                excitatory_spikes=surround_spikes, inhibitory_spikes=np.array([center_spike]),
                weights_exc=off_exc_weights, weights_inh=off_inh_weights, 
                V_thresh=V_thresh, **neuron_params)
    return on_spike_times, off_spike_times

def runThresholdExperiment(input_image):
    print("--- Running Threshold Experiment ---")
    visualize_image(input_image, "Input Image")
    print("1. Layer 0: Encoding image to photoreceptor spikes...")
    photoreceptor_spikes = intensityToDelayEncoding(input_image)
    thresholds_to_test = [20.0, 15.0, 10.0, 5.0]
    print("\n2. Layer 1: Processing spikes with different neuron thresholds...")
    for thresh in thresholds_to_test:
        print(f"   - Testing threshold: {thresh:.1f} mV")
        on_spikes, off_spikes = applyReceptiveFieldsLIF(
            photoreceptor_spikes, V_thresh=thresh
        )
        all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
        visualize_spike_times(all_edges_spikes, f"Combined Edges (Threshold = {thresh:.1f} mV)")

def runOnOffPipeline(input_image, title):
    print(f"\n--- Processing with ON/OFF receptive fields: {title} ---")
    visualize_image(input_image, f"Input Image: {title}")
    on_filter, off_filter = create_on_off_filters(center_weight=8, surround_weight=-1)
    print("Applying ON/OFF receptive fields...")
    on_spikes, off_spikes = apply_receptive_fields(input_image, on_filter, off_filter)
    print("Visualizing ON-cell and OFF-cell spike times...")
    visualize_spike_times(on_spikes, "ON-Center Cell Spike Times")
    visualize_spike_times(off_spikes, "OFF-Center Cell Spike Times")
    visualize_static_raster_plot(on_spikes, "ON-Center Raster Plot")
    visualize_static_raster_plot(off_spikes, "OFF-Center Raster Plot")
    print("Combining channels with Winner-Take-All to find all edges...")
    all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
    visualize_spike_times(all_edges_spikes, "Combined Edges (Complex Cell Simulation)")

def runOnOffPipelineLeaky(input_image, title):
    print(f"\n--- Processing with ON/OFF receptive fields: {title} ---")
    visualize_image(input_image, f"Input Image: {title}")
    on_filter, off_filter = create_on_off_filters(center_weight=8, surround_weight=-1)
    print("Applying ON/OFF receptive fields...")
    on_spikes, off_spikes = apply_receptive_fields_with_leaky_bucket(input_image)
    print("Visualizing ON-cell and OFF-cell spike times...")
    visualize_spike_times(on_spikes, "ON-Center Cell Spike Times")
    visualize_spike_times(off_spikes, "OFF-Center Cell Spike Times")
    visualize_static_raster_plot(on_spikes, "ON-Center Raster Plot")
    visualize_static_raster_plot(off_spikes, "OFF-Center Raster Plot")
    print("Combining channels with Winner-Take-All to find all edges...")
    all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
    visualize_spike_times(all_edges_spikes, "Combined Edges (Complex Cell Simulation)")

def experiment_2(image, filter_pos, filter_neg, threshold=3):
    height, width = image.shape
    output_height = height - 2
    output_width = width - 2
    firing_times = np.full((output_height, output_width), np.nan)
    pos_spikes = intensity_to_delay_encoding(image)
    neg_spikes = negative_image_encoding(image)
    for y in range(output_height):
        for x in range(output_width):
            pos_patch = pos_spikes[y:y+3, x:x+3]
            neg_patch = neg_spikes[y:y+3, x:x+3]
            synapses = []
            for i in range(3):
                for j in range(3):
                    if filter_pos[i,j] == 1:
                        synapses.append(pos_patch[i, j])
                    if filter_neg[i,j] == 1:
                        synapses.append(neg_patch[i, j])
            firing_time = integrate_and_fire(np.array(synapses), None, threshold)
            if firing_time is not None:
                firing_times[y, x] = firing_time
    return firing_times


def run_on_off_pipeline_with_if(input_image, title, threshold=1.0):
    print(f"\n--- Spike-Driven ON/OFF Pipeline: {title} (Threshold={threshold}) ---")
    visualize_image(input_image, f"Input Image: {title}")
    print("Layer 0: Encoding image to photoreceptor spikes...")
    photoreceptor_spikes = intensity_to_delay_encoding(input_image)
    visualize_spike_times(photoreceptor_spikes, "Layer 0: Photoreceptor Spike Times")
    print("Layer 1: Applying ON/OFF receptive fields with Integrate-and-Fire neurons...")
    on_spikes, off_spikes = apply_receptive_fields_with_if(photoreceptor_spikes, threshold=threshold)
    print("Visualizing ON-cell and OFF-cell spike times...")
    visualize_spike_times(on_spikes, f"Layer 1: ON-Center Cell Spikes (Thresh={threshold})")
    visualize_spike_times(off_spikes, f"Layer 1: OFF-Center Cell Spikes (Thresh={threshold})")
    print("Combining channels to find all edges...")
    all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
    visualize_spike_times(all_edges_spikes, "Combined Edges")

def runPipeline(input_image, title):
    print(f"\n--- Processing: {title} ---")
    visualize_image(input_image, f"Input Image: {title}")
    filter_pos_v_edge = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    filter_neg_v_edge = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    filter_pos_h_edge = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    filter_neg_h_edge = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    filter_pos_diag1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    filter_neg_diag1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    filter_pos_diag2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    filter_neg_diag2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print("Applying all filters...")
    pos_spikes = intensity_to_delay_encoding(input_image)
    neg_spikes = negative_image_encoding(input_image)
    visualize_spike_times(pos_spikes, f"pos")
    visualize_spike_times(neg_spikes, f"neg")
    visualizeRasterPlot(pos_spikes, f"pos")
    visualizeRasterPlot(neg_spikes, f"neg")
    v_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_v_edge, filter_neg_v_edge, threshold=6)
    v_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_v_edge, filter_pos_v_edge, threshold=6)
    h_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_h_edge, filter_neg_h_edge, threshold=6)
    h_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_h_edge, filter_pos_h_edge, threshold=6)
    d1_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_diag1, filter_neg_diag1, threshold=6)
    d1_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_diag1, filter_pos_diag1, threshold=6)
    d2_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_diag2, filter_neg_diag2, threshold=6)
    d2_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_diag2, filter_pos_diag2, threshold=6)
    visualize_spike_times(v_pos_edge_times, f"v_pos_edge")
    visualize_spike_times(h_pos_edge_times, f"h_pos_edge")
    visualize_spike_times(d1_pos_edge_times, f"d1_pos_edge")
    visualize_spike_times(d2_pos_edge_times, f"d2_pos_edge")
    print("Performing location-wise Winner-Take-All...")
    location_wta_output = location_wise_wta([v_pos_edge_times, v_neg_edge_times,
                                             h_pos_edge_times, h_neg_edge_times,
                                             d1_pos_edge_times, d1_neg_edge_times,
                                             d2_pos_edge_times, d2_neg_edge_times])
    visualize_spike_times(location_wta_output, f"edge")

if __name__ == "__main__":
    # image_path = 'data/doomguy.jpg'
    image_path = 'data/lenna.png'
    original_image = Image.open(image_path)
    grayscale_image = original_image.convert('L')
    max_dim = 256
    low_res_image_pil = grayscale_image.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    input_image = np.array(low_res_image_pil)
    # input_image = generate_checkerboard(size=256, block_size=32)
    # run_pipeline(input_image, "edge detection")
    # The threshold is now a key parameter to tune!
    # A low threshold makes the neurons more sensitive.
    # run_on_off_pipeline_with_if(input_image, "ON/OFF Edge Detection", threshold=1.0)
    # run_lif_pipeline(input_image, "LIF Edge Detection")
    # Define common weights and neuron params
    W_CENTER_EXC = 60.0 # Increased weight for this model
    W_SURROUND_INH = -7.5 # (8 * -7.5 = -60.0, balanced)
    neuron_params = {
        'weights_exc': np.array([W_CENTER_EXC]),
        'weights_inh': np.full(8, W_SURROUND_INH),
        'tau_syn_exc': 5.0, 'tau_syn_inh': 8.0, # Inhibitory current lasts a bit longer
        'tau_m': 10.0, 'V_thresh': 15.0
    }
    
    print("--- Demonstrating Neuron Dynamics ---")
    
    # Scenario 1: High Contrast Edge (ON-cell fires)
    # Excitatory spike is early (bright center), inhibitory spikes are late (dark surround)
    time, V, Ie, Ii = leakyIntegrateAndFire(excitatory_spikes=np.array([10.0]), # Early exc spike
        inhibitory_spikes=np.full(8, 80.0), # Late inh spikes
        **neuron_params, return_trace=True)

    plotNeuronVoltageTrace("Scenario 1: High Contrast (Neuron Fires Early)", time, V, Ie, Ii)

    # Scenario 2: Uniform Bright Field (ON-cell is suppressed)
    # All spikes arrive early

    time, V, Ie, Ii = leakyIntegrateAndFire(excitatory_spikes=np.array([10.0]), # Early exc spike
        inhibitory_spikes=np.full(8, 10.0), # Early inh spikes
        **neuron_params, return_trace=True), # Early exc spike
    plotNeuronVoltageTrace("Scenario 2: Uniform Field (Neuron Is Suppressed)", time, V, Ie, Ii )

    runThresholdExperiment(input_image)
    # run_on_off_pipeline_leaky(input_image, "ON/OFF Edge Detection")
