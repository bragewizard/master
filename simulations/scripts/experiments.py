import numpy as np

from visualize import visualize_3d_spikes, visualize_raster_plot, visualize_sequence_raster_plot,\
visualize_image, visualize_spike_times, visualize_static_raster_plot, visualize_spikes,\
visualize_stimuli_animation,plot_neuron_voltage_trace
from core import intensity_to_delay_encoding

def threshold_experiment(input_image):
    print("--- Running Threshold Experiment ---")
    visualize_image(input_image, "Input Image")
    print("1. Layer 0: Encoding image to photoreceptor spikes...")
    photoreceptor_spikes = intensity_to_delay_encoding(input_image)
    thresholds_to_test = [20.0, 15.0, 10.0, 5.0]
    print("\n2. Layer 1: Processing spikes with different neuron thresholds...")
    for thresh in thresholds_to_test:
        print(f"   - Testing threshold: {thresh:.1f} mV")
        on_spikes, off_spikes = calculate_first_layer_lif(
            photoreceptor_spikes, V_thresh=thresh
        )
        all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
        visualize_spike_times(all_edges_spikes, f"Combined Edges (Threshold = {thresh:.1f} mV)")

def on_off_pipeline(input_image, title):
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

def lif_on_off_pipeline(input_image, title):
    print(f"\n--- Processing with ON/OFF receptive fields: {title} ---")
    visualize_image(input_image, f"Input Image: {title}")
    on_filter, off_filter = create_on_off_filters(center_weight=8, surround_weight=-1)
    print("Applying ON/OFF receptive fields...")
    on_spikes, off_spikes = apply_receptive_fields_lif(input_image)
    print("Visualizing ON-cell and OFF-cell spike times...")
    visualize_spike_times(on_spikes, "ON-Center Cell Spike Times")
    visualize_spike_times(off_spikes, "OFF-Center Cell Spike Times")
    visualize_static_raster_plot(on_spikes, "ON-Center Raster Plot")
    visualize_static_raster_plot(off_spikes, "OFF-Center Raster Plot")
    print("Combining channels with Winner-Take-All to find all edges...")
    all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
    visualize_spike_times(all_edges_spikes, "Combined Edges (Complex Cell Simulation)")

def if_on_off_pipeline(input_image, title, threshold=1.0):
    print(f"\n--- Spike-Driven ON/OFF Pipeline: {title} (Threshold={threshold}) ---")
    visualize_image(input_image, f"Input Image: {title}")
    print("Layer 0: Encoding image to photoreceptor spikes...")
    photoreceptor_spikes = intensity_to_delay_encoding(input_image)
    visualize_spike_times(photoreceptor_spikes, "Layer 0: Photoreceptor Spike Times")
    print("Layer 1: Applying ON/OFF receptive fields with Integrate-and-Fire neurons...")
    on_spikes, off_spikes = apply_receptive_fields(photoreceptor_spikes)
    print("Visualizing ON-cell and OFF-cell spike times...")
    visualize_spike_times(on_spikes, f"Layer 1: ON-Center Cell Spikes (Thresh={threshold})")
    visualize_spike_times(off_spikes, f"Layer 1: OFF-Center Cell Spikes (Thresh={threshold})")
    print("Combining channels to find all edges...")
    all_edges_spikes = location_wise_wta([on_spikes, off_spikes])
    visualize_spike_times(all_edges_spikes, "Combined Edges")

def pipeline(input_image):
    visualize_image(input_image, f"Image")
    spikes = intensity_to_delay_encoding(input_image)
    print(spikes)
    height, width = spikes.shape
    y_coords, x_coords = np.where(~np.isnan(spikes))
    times = spikes[y_coords, x_coords]
    neuron_indices = y_coords * width + x_coords

def visualize_rendom_stimuli():
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
    visualize_sequence_raster_plot(stimuli, frame_duration=100)
