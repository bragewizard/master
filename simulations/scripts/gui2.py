from math import exp
import numpy as np
import threading
import dearpygui.dearpygui as dpg
import time
import bisect

from theme import create_global_theme, create_line_series_theme, load_font, scatter_green, scatter_orange
from core import intensity_to_delay_encoding, create_conv_connections, get_postsynaptic_events

# --- Simulation Parameters ---
INPUT_SHAPE = (28, 28)
OUTPUT_SHAPE = (26, 26)
OUTPUT_WIDTH = OUTPUT_SHAPE[1]
NUM_NEURONS_PER_MAP = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
TOTAL_NEURONS = NUM_NEURONS_PER_MAP * 2

# LIF Neuron Parameters
TAU_M = 10.0          # Membrane time constant (ms)
V_THRESH = 1.0        # Firing threshold
V_REST = 0.0          # Resting potential
W_LATERAL_INH = -0.5  # Weight of lateral inhibition
LATERAL_DELAY = 1.0   # Delay for lateral inhibition spike (ms)

# --- Global State for Simulation Control ---
is_paused = True
step_event = threading.Event()
simulation_time = 0.0

# --- Helper Functions ---
def scale_image(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)

def key_press_handler(sender, app_data):
    """Handles keyboard inputs for pausing and stepping the simulation."""
    global is_paused
    key = app_data
    if key == 524:
        is_paused = not is_paused
        print(f"Simulation {'Paused' if is_paused else 'Playing'}")
    elif key == 559 and is_paused:
        step_event.set()

# --- The New LIF Simulation Engine ---
def run_lif_simulation(spikes, indices):
    global simulation_time, is_paused
    connection_map = create_conv_connections(INPUT_SHAPE, OUTPUT_SHAPE)
    arrival_times, target_indices, weights = get_postsynaptic_events(spikes, indices, connection_map)

    # Event Queue: (time, type, data_dict)
    # Types: 'FF' (Feed-Forward), 'LI' (Lateral Inhibition)
    event_queue = []
    for t, target, w in zip(arrival_times, target_indices, weights):
        event_queue.append((t, 'FF', {'target': target, 'weight': w}))
    event_queue.sort(key=lambda x: x[0])

    # Neuron state variables
    membrane_potentials = np.full(TOTAL_NEURONS, V_REST, dtype=float)
    last_update_times = np.zeros(TOTAL_NEURONS, dtype=float)
    output_spike_times = np.full(TOTAL_NEURONS, np.nan, dtype=float)
    
    # Output image state
    output_image = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
    output_spike_list_x, output_spike_list_y, output_spike_list_idx = [], [], []

    # 2. MAIN SIMULATION LOOP (Event-Driven)
    while event_queue:
        if is_paused:
            step_event.wait()  # Block here until spacebar or right arrow
            step_event.clear()

        event_time, event_type, data = event_queue.pop(0)
        simulation_time = event_time
        target_idx = data['target']
        weight = data['weight']

        # Skip processing if neuron has already fired
        if not np.isnan(output_spike_times[target_idx]):
            continue

        # 3. LIF DYNAMICS: Decay and Integrate
        time_delta = event_time - last_update_times[target_idx]
        if time_delta > 0:
            membrane_potentials[target_idx] *= exp(-time_delta / TAU_M)
        
        membrane_potentials[target_idx] += weight
        last_update_times[target_idx] = event_time

        # 4. CHECK FOR FIRING
        if membrane_potentials[target_idx] >= V_THRESH:
            # --- A neuron fired! ---
            output_spike_times[target_idx] = event_time
            membrane_potentials[target_idx] = V_REST # Reset potential

            # Add to our list for the raster plot
            output_spike_list_x.append(event_time)
            output_spike_list_y.append(target_idx)

            # 5. UPDATE DYNAMIC OUTPUT IMAGE
            # Map 1D index back to 2D image coordinate
            map_offset = 0
            if target_idx >= NUM_NEURONS_PER_MAP: # Check if it's an OFF neuron
                map_offset = NUM_NEURONS_PER_MAP
                
            neuron_2d_idx = target_idx - map_offset
            y = neuron_2d_idx // OUTPUT_WIDTH
            x = neuron_2d_idx % OUTPUT_WIDTH
            
            # Set pixel brightness based on how early it fired (earlier = brighter)
            max_time = spikes[-1] if len(spikes) > 0 else 1.0
            brightness = max(0.1, 1.0 - (event_time / max_time))
            output_image[y, x] = brightness
            
            # 6. GENERATE LATERAL INHIBITION EVENTS
            # Inhibit immediate neighbors in the same map (ON or OFF)
            neighbors = [target_idx - 1, target_idx + 1]
            for neighbor_idx in neighbors:
                # Check bounds and ensure neighbor is on the same row
                if 0 <= neighbor_idx < TOTAL_NEURONS and (neighbor_idx // OUTPUT_WIDTH) == (target_idx // OUTPUT_WIDTH):
                    inhib_event = (event_time + LATERAL_DELAY, 'LI', {'target': neighbor_idx, 'weight': W_LATERAL_INH})
                    # Use bisect to insert event while keeping the queue sorted
                    bisect.insort(event_queue, inhib_event, key=lambda x: x[0])

        # 7. UPDATE THE GUI
        # Only update GUI periodically to avoid lag
        if len(output_spike_list_x) % 10 == 0 or is_paused:
            dpg.set_value("out_spikes_series", [output_spike_list_x, output_spike_list_y])

            # Update the dynamic texture
            scaled_out_img = scale_image(output_image, int(256/OUTPUT_SHAPE[0]))
            h, w = scaled_out_img.shape
            rgba_image = np.zeros((h, w, 4), dtype=np.float32)
            rgba_image[..., :3] = scaled_out_img[..., np.newaxis]
            rgba_image[..., 3] = 1.0
            dpg.set_value("output_texture", rgba_image.flatten())

    print("Simulation Finished.")
    is_paused = True # Pause at the end

# --- GUI Layout ---
VIEWPORT_WIDTH = 2300
VIEWPORT_HEIGHT = 1300

def run_gui(input_image):
    spikes = intensity_to_delay_encoding(input_image)
    y_coords, x_coords = np.where(~np.isnan(spikes))
    times = spikes[y_coords, x_coords]
    neuron_indices = y_coords * INPUT_SHAPE[1] + x_coords
    sorted_idx = np.argsort(times)
    spikes, indices = times[sorted_idx], neuron_indices[sorted_idx]

    dpg.create_context()
    load_font(dpg)
    
    # --- Register the keyboard handler ---
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=key_press_handler)

    with dpg.window(tag="primary_window",
                    pos=[0, 0],
                    width=VIEWPORT_WIDTH,
                    height=VIEWPORT_HEIGHT,
                    no_move=True,
                    no_resize=True,
                    no_close=True,
                    no_collapse=True,
                    no_title_bar=True):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=VIEWPORT_WIDTH * 0.7):
                with dpg.plot(label="LIF Layer Output Spikes", height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (ms)")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Neuron Index", tag="y_axis_lif")
                    dpg.set_axis_limits("y_axis_lif", 0, TOTAL_NEURONS)
                    dpg.add_scatter_series(x=[], y=[], label="Output Spike", parent="y_axis_lif", tag="out_spikes_series")

            with dpg.child_window(width=-1):
                dpg.add_text("Input Image")
                # Setup for original input image
                SCALE_FACTOR_IN = int(512 / INPUT_SHAPE[0])
                normalized_image = input_image.astype(np.float32) / 255.0
                scaled_in_img = scale_image(normalized_image, SCALE_FACTOR_IN)
                h, w = scaled_in_img.shape
                rgba_in = np.zeros((h, w, 4), dtype=np.float32); rgba_in[..., :3] = scaled_in_img[..., np.newaxis]; rgba_in[..., 3] = 1.0
                
                dpg.add_text("LIF Layer Output (Time-to-first-spike)")
                # Setup for dynamic output image
                SCALE_FACTOR_OUT = int(512 / OUTPUT_SHAPE[0])
                h_out, w_out = OUTPUT_SHAPE[0] * SCALE_FACTOR_OUT, OUTPUT_SHAPE[1] * SCALE_FACTOR_OUT
                
                with dpg.texture_registry():
                    dpg.add_static_texture(width=w, height=h, default_value=rgba_in.flatten(), tag="input_texture")
                    dpg.add_dynamic_texture(width=w_out, height=h_out, default_value=np.zeros(w_out*h_out*4, dtype=np.float32), tag="output_texture")

                dpg.add_image("input_texture")
                dpg.add_image("output_texture")
                dpg.add_separator()
                dpg.add_text("Controls: [Space] to Play/Pause, [n] to Step")
                dpg.add_text(f"Frame: {dpg.get_frame_count()}", tag="frame_text")

    dpg.create_viewport(title='SNN', width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT)
    dpg.setup_dearpygui()
    
    update_thread = threading.Thread(target=run_lif_simulation, args=(spikes, indices), daemon=True)
    update_thread.start()
    
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
