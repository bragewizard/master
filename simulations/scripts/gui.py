from math import sin, exp, pi
import numpy as np
import random
import threading
import dearpygui.dearpygui as dpg
import time
import collections

from theme import create_global_theme, create_line_series_theme, load_font, scatter_green, scatter_orange
from core import intensity_to_delay_encoding,create_conv_connections, get_postsynaptic_events

MAX_POINTS = 2000
TIME_STEP = 0.05
# INPUT_SHAPE = (28, 28)
# OUTPUT_SHAPE = (26, 26)
INPUT_SHAPE = (16, 16)
OUTPUT_SHAPE = (14, 14)
TOTAL_OUTPUT_NEURONS = (14*14) * 2

connection_map = create_conv_connections(INPUT_SHAPE, OUTPUT_SHAPE)
start_time = -(MAX_POINTS - 1) * TIME_STEP
initial_time_data = np.linspace(start_time, 0.0, num=MAX_POINTS)
time_data = collections.deque(initial_time_data, maxlen=MAX_POINTS)
y_data = collections.deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
incoming_spikes = collections.deque([0,0] * MAX_POINTS, maxlen=MAX_POINTS)
simulation_time = 0
is_running = True

def scale_image(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)

def generate_brain_signal(t, base_freq=10.0):
    y1 = 1.0 * sin(2 * pi * base_freq * t)
    y2 = 0.5 * sin(2 * pi * (base_freq * 1.8) * t + 0.5)
    y3 = 0.8 * sin(2 * pi * (base_freq * 0.4) * t + 1.2)
    noise = (random.random() - 0.5) * 0.8
    return (y1 + y2 + y3) / 2.3 + noise

def update_frame_counter():
    dpg.set_value("frame_text", f"Frame: {dpg.get_frame_count()}")

def update_series_data(spikes, indices):
    global simulation_time

    sorted_indices = np.argsort(spikes)
    spikes = spikes[sorted_indices]
    indices = indices[sorted_indices]
    arrival_times, target_indices, weights = get_postsynaptic_events(
        input_spike_times=spikes,
        input_spike_indices=indices,
        connections=connection_map
    )
    TARGET_Y = 3
    TARGET_X = 3
    OUTPUT_WIDTH = 14
    target_neuron_idx = (OUTPUT_WIDTH * OUTPUT_WIDTH) + (TARGET_Y * OUTPUT_WIDTH + TARGET_X)
    print(f"🔬 Monitoring Neuron Index: {target_neuron_idx}")
    mask = (target_indices == target_neuron_idx)
    neuron_input_times = arrival_times[mask]
    neuron_input_weights = weights[mask]
    print(f"✅ Neuron {target_neuron_idx} received {len(neuron_input_times)} synaptic inputs.")
    while True:
        update_frame_counter()
        new_y_value = generate_brain_signal(simulation_time)
        w_exc = dpg.get_value("w_exc_slider")
        w_inh = dpg.get_value("w_inh_slider")
        I_exc = 0.0
        I_inh = 0.0
        
        y_data.append(new_y_value)
        time_data.append(simulation_time)

        exc_mask = weights > 0
        exc_times = arrival_times[exc_mask]
        exc_indices = target_indices[exc_mask]
        inh_mask = weights < 0
        inh_times = arrival_times[inh_mask]
        inh_indices = target_indices[inh_mask]

        # spike_times_exc = [s[0] for s in incoming_spikes if s[1] == 1 and time_data[0] <= s[0] <= time_data[-1]]
        # spike_times_inh = [s[0] for s in incoming_spikes if s[1] == -1 and time_data[0] <= s[0] <= time_data[-1]]
        # spike_times_out = [s[0] for s in incoming_spikes if s[1] == 2 and time_data[0] <= s[0] <= time_data[-1]]

        dpg.set_value('series_tag', [list(time_data), list(y_data)])
        # dpg.set_value('exc_spikes_series', [list(neuron_input_times), list(neuron_input_weights)])
        dpg.set_value('exc_spikes_series', [list(exc_times), list(exc_indices)])
        dpg.set_value('inh_spikes_series', [list(inh_times), list(inh_indices)])
        # dpg.set_value('out_spikes_series',[[t for t in spike_times_out], [0 for _ in spike_times_out]])
        dpg.set_axis_limits("x_axis", time_data[0], time_data[-1])
        dpg.set_axis_limits("x_axis2", time_data[0], time_data[-1])
        simulation_time += TIME_STEP
        time.sleep(0.01)

def reset_simulation():
    global simulation_time, incoming_spikes
    simulation_time = 0.0
    incoming_spikes.clear()
    time_data.clear()
    y_data.clear()
    time_data.extend(initial_time_data)
    y_data.extend([0.0] * MAX_POINTS)
    print("Simulation Reset")

def inject_excitatory_spike():
    incoming_spikes.append((simulation_time, 1))

def inject_inhibitory_spike():
    incoming_spikes.append((simulation_time, -1))

# --- GUI Layout ---
VIEWPORT_WIDTH = 2300
VIEWPORT_HEIGHT = 1300

def run_gui(input_image):

    spikes = intensity_to_delay_encoding(input_image)
    height, width = spikes.shape
    y_coords, x_coords = np.where(~np.isnan(spikes))
    times = spikes[y_coords, x_coords]
    neuron_indices = y_coords * width + x_coords

    sorted_indices = np.argsort(times)
    spikes = times[sorted_indices]
    indices = neuron_indices[sorted_indices]
    dpg.create_context()

    load_font(dpg)
    plot_theme = create_line_series_theme(dpg)
    green = scatter_green(dpg)
    orange = scatter_orange(dpg)

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
                with dpg.plot(label="Neuron Membrane Potential", height=VIEWPORT_HEIGHT // 2 - 40, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Displacement", tag="y_axis")
                    dpg.set_axis_limits("y_axis", 0, TOTAL_OUTPUT_NEURONS)
                    # dpg.set_axis_limits("y_axis", -1.2, 1.2)
                    dpg.add_scatter_series(x=[], y=[], label="Inhibitory Spike", parent="y_axis", tag="inh_spikes_series")
                    dpg.add_scatter_series(x=[], y=[], label="Excitatory Spike", parent="y_axis", tag="exc_spikes_series")
                    dpg.add_scatter_series(x=[], y=[], label="Output Spike", parent="y_axis", tag="out_spikes_series")
                    # dpg.bind_item_theme("inh_spikes_series", orange)
                    # dpg.bind_item_theme("out_spikes_series", orange)
                    dpg.bind_item_theme("exc_spikes_series", green)

                
                # Current Plot
                with dpg.plot(label="Synaptic Currents", height=VIEWPORT_HEIGHT // 2 - 40, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (ms)", tag="x_axis2")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Displacement", tag="y_axis2")
                    dpg.set_axis_limits("y_axis2", -5, 5)
                    dpg.add_line_series(list(time_data), list(y_data), label="Excitatory Current", parent="y_axis2", tag="series_tag")
                    dpg.bind_item_theme("series_tag", plot_theme)

            with dpg.child_window(width=-1):
                dpg.add_text("Neuron Controls")
                dpg.add_separator()
                dpg.add_spacer(height=10)
                dpg.add_button(label="Inject Excitatory Spike (+)", callback=inject_excitatory_spike, width=-1, height=40)
                dpg.add_spacer(height=10)
                dpg.add_button(label="Inject Inhibitory Spike (-)", callback=inject_inhibitory_spike, width=-1, height=40)
                dpg.add_spacer(height=20)
                dpg.add_slider_float(label="Excitatory Weight",
                                     tag="w_exc_slider",
                                     default_value=30.0,
                                     min_value=0.0,
                                     max_value=100.0,
                                     width=-200)
                dpg.add_slider_float(label="Inhibitory Weight",
                                     tag="w_inh_slider",
                                     default_value=-15.0,
                                     min_value=-100.0,
                                     max_value=0.0,
                                     width=-200)
                dpg.add_spacer(height=20)
                dpg.add_button(label="Reset Simulation", callback=reset_simulation, width=-1, height=40)
                dpg.add_separator()

                SCALE_FACTOR = int(256 / INPUT_SHAPE[0])
                normalized_image = input_image.astype(np.float32) / 255.0
                scaled_image = scale_image(normalized_image, SCALE_FACTOR)
                scaled_height, scaled_width = scaled_image.shape
                rgba_image = np.zeros((scaled_height, scaled_width, 4), dtype=np.float32)
                rgba_image[..., :3] = scaled_image[..., np.newaxis]
                rgba_image[..., 3] = 1.0
                texture_data = rgba_image.flatten()

                with dpg.texture_registry(show=True):
                    dpg.add_static_texture(
                        width=scaled_width,
                        height=scaled_height,
                        default_value=texture_data,
                        tag="texture_tag"
                    )

                dpg.add_image("texture_tag")
                # dpg.add_separator()
                dpg.add_text(f"Frame: {dpg.get_frame_count()}", tag="frame_text")

    dpg.create_viewport(title='Neuron Simulation', width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT)
    dpg.set_viewport_vsync(True)
    dpg.setup_dearpygui()
    update_thread = threading.Thread(target=update_series_data, args=(spikes, indices), daemon=True)
    update_thread.start()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
