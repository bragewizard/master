from math import sin, exp, pi
import numpy as np
import random
import threading
import dearpygui.dearpygui as dpg
import time
import collections
import bisect

from theme import (
    create_global_theme,
    create_line_series_theme,
    load_font,
    scatter_green,
    scatter_orange,
)
from core import (
    intensity_to_delay_encoding,
    create_conv_connections,
    get_postsynaptic_events,
)
from data import scale_image, rgba_image

MAX_POINTS = 2000
TIME_STEP = 0.01
# INPUT_SHAPE = (28, 28)
# OUTPUT_SHAPE = (28, 28)
INPUT_SHAPE = (8, 8)
INPUT_NEURONS = INPUT_SHAPE[0] * INPUT_SHAPE[1]
OUTPUT_SHAPE = (6, 6)
OUTPUT_WIDTH = OUTPUT_SHAPE[1]
INPUT_WIDTH = INPUT_SHAPE[1]
NUM_NEURONS_PER_MAP = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
TOTAL_NEURONS = NUM_NEURONS_PER_MAP
V_REST = 0
V_TRESH = 4
TAU_M = 1
SCALE_FACTOR_1 = int(256 / INPUT_SHAPE[0])
SCALE_FACTOR_2 = int(256 / OUTPUT_SHAPE[0])

is_paused = False
step_event = threading.Event()
simulation_time = 0.0

start_time = -(MAX_POINTS - 1) * TIME_STEP
initial_time_data = np.linspace(start_time, 0.0, num=MAX_POINTS)
time_data = collections.deque(initial_time_data, maxlen=MAX_POINTS)
simulation_time = 0
neurons_viz = None
connections_viz = None

# NETWORK STATE
connections = {}  # key: neuron index, val: (neuron indices, weights)
potentials = np.full(TOTAL_NEURONS, V_REST, dtype=float)


def key_press_handler(sender, app_data):
    global is_paused
    key = app_data
    if key == 524:
        is_paused = not is_paused
        if not is_paused:
            step_event.set()
        print(f"Simulation {'Paused' if is_paused else 'Playing'}")
    elif key == 559 and is_paused:
        step_event.set()


def update_text():
    global simulation_time
    while True:
        dpg.set_value("frame_text", f"Frame: {dpg.get_frame_count()}")
        dpg.set_value("time_text", f"Time: {simulation_time}")
        time.sleep(0.1)


def fill_event_queue(event_queue, arrival_times, target_indices):
    for t, target in zip(arrival_times, target_indices):
        event_queue.append((t, "FF", {"target": target}))
    event_queue.sort(key=lambda x: x[0])


def update_series_data(data: np.ndarray):
    global simulation_time, neurons_viz, connections_viz, potentials, connections

    for iteration in range(3):
        input_image = data[iteration]
        normalized_image = input_image.astype(np.float32) / 255.0
        scaled_image = scale_image(normalized_image, SCALE_FACTOR_1)
        scaled_image_rgba = rgba_image(scaled_image).flatten()
        dpg.set_value("input_image", scaled_image_rgba)

        spikes = intensity_to_delay_encoding(input_image)
        height, width = spikes.shape
        y_coords, x_coords = np.where(~np.isnan(spikes))
        times = spikes[y_coords, x_coords]
        neuron_indices = y_coords * width + x_coords
        sorted_indices = np.argsort(times)
        spikes = times[sorted_indices]
        indices = neuron_indices[sorted_indices]

        sorted_indices = np.argsort(spikes)
        spikes = spikes[sorted_indices] + simulation_time
        indices = indices[sorted_indices]

        event_queue = []
        fill_event_queue(event_queue, spikes, indices)
        last_update_times = np.zeros(INPUT_NEURONS, dtype=float)
        output_spike_times = np.full(TOTAL_NEURONS, np.nan, dtype=float)
        out_img = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        out_img = rgba_image(out_img)
        in_img = np.zeros(INPUT_SHAPE, dtype=np.float32)
        in_img = rgba_image(in_img)
        output_spike_list_x, output_spike_list_y, output_spike_list_idx = [], [], []

        event_time, event_type, event_data = event_queue.pop(0)
        now_time = event_time

        while len(event_queue) > 0:
            if is_paused:
                step_event.wait()
                step_event.clear()

            while now_time < simulation_time:
                target_idx = event_data["target"]
                # weight = event_data['weight']

                # if not np.isnan(output_spike_times[target_idx]):
                #     continue

                time_delta = event_time - last_update_times[target_idx]
                # if time_delta > 0:
                # potentials[target_idx] *= exp(-time_delta / TAU_M)

                # potentials[target_idx] += weight
                last_update_times[target_idx] = event_time

                # if potentials[target_idx] >= V_THRESH:
                # output_spike_times[target_idx] = event_time
                # potentials[target_idx] = V_REST
                # output_spike_list_x.append(event_time)
                # output_spike_list_y.append(target_idx)
                # map_offset = 0
                # if target_idx >= NUM_NEURONS_PER_MAP:
                # map_offset = NUM_NEURONS_PER_MAP

                neuron_2d_idx = target_idx
                y = neuron_2d_idx // INPUT_WIDTH
                x = neuron_2d_idx % INPUT_WIDTH
                brightness = 1
                in_img[y, x, 0] = brightness
                in_img[y, x, 1] = brightness
                in_img[y, x, 2] = brightness
                in_img[y, x, 3] = 1
                event_time, event_type, event_data = event_queue.pop(0)
                now_time = event_time

            simulation_time += TIME_STEP
            time_data.append(simulation_time)
            time.sleep(0.001)
            in_img[:, :, 0:3] *= 0.996
            out_img[:, :, 0:3] *= 0.996
            scaled_img = scale_image(in_img, SCALE_FACTOR_1)
            scaled_img_out = scale_image(out_img, SCALE_FACTOR_2)
            dpg.set_value("neurons0", scaled_img.flatten())
            dpg.set_value("neurons1", scaled_img_out.flatten())
            dpg.configure_item(item=neurons_viz, points=potentials)


def reset_simulation():
    global simulation_time
    simulation_time = 0.0
    time_data.clear()
    time_data.extend(initial_time_data)
    print("Simulation Reset")


def inject_excitatory_spike():
    incoming_spikes.append((simulation_time, 1))


def inject_inhibitory_spike():
    incoming_spikes.append((simulation_time, -1))


# --- GUI Layout ---
VIEWPORT_WIDTH = 2300
VIEWPORT_HEIGHT = 1300


def run_gui(data: np.ndarray):
    dpg.create_context()

    load_font(dpg)

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=key_press_handler)

    with dpg.window(
        tag="primary_window",
        pos=[0, 0],
        width=VIEWPORT_WIDTH,
        height=VIEWPORT_HEIGHT,
        no_move=True,
        no_resize=True,
        no_close=True,
        no_collapse=True,
        no_title_bar=True,
    ):
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                with dpg.child_window(
                    width=VIEWPORT_WIDTH * 0.7, height=VIEWPORT_HEIGHT * 0.21
                ):
                    default_value1 = np.zeros(INPUT_SHAPE, dtype=np.float32)
                    default_value = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
                    default_image1 = scale_image(default_value1, SCALE_FACTOR_1)
                    default_image = scale_image(default_value, SCALE_FACTOR_2)
                    default_image_rgba = rgba_image(default_image).flatten()
                    default_image1_rgba = rgba_image(default_image1).flatten()
                    with dpg.texture_registry(show=True):
                        dpg.add_dynamic_texture(
                            width=default_image1.shape[1],
                            height=default_image1.shape[0],
                            default_value=default_image1_rgba,
                            tag="neurons0",
                        )
                        dpg.add_dynamic_texture(
                            width=default_image.shape[1],
                            height=default_image.shape[0],
                            default_value=default_image_rgba,
                            tag="neurons1",
                        )

                    with dpg.group(horizontal=True):
                        dpg.add_image("neurons0")
                        dpg.add_image("neurons1")

                with dpg.child_window(
                    width=VIEWPORT_WIDTH * 0.7, height=VIEWPORT_HEIGHT * 0.77
                ):
                    with dpg.drawlist(width=300, height=1980):
                        for i in range(INPUT_SHAPE[0] * INPUT_SHAPE[1]):
                            dpg.draw_circle((5, i * 8 + 15), 3, fill=(255, 255, 255))
                        for i in range(OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]):
                            dpg.draw_circle((200, i * 8 + 115), 3, fill=(255, 255, 255))

            with dpg.child_window(width=-1):
                dpg.add_button(
                    label="Reset Simulation",
                    callback=reset_simulation,
                    width=-1,
                    height=40,
                )
                dpg.add_separator()

                normalized_image = data[0].astype(np.float32) / 255.0
                input_image = scale_image(normalized_image, SCALE_FACTOR_1)
                input_image_rgba = rgba_image(input_image).flatten()

                with dpg.texture_registry(show=True):
                    dpg.add_dynamic_texture(
                        width=input_image.shape[1],
                        height=input_image.shape[0],
                        default_value=input_image_rgba,
                        tag="input_image",
                    )
                with dpg.group(horizontal=True):
                    dpg.add_image("input_image")
                    with dpg.drawlist(width=30, height=100):
                        dpg.draw_line(
                            (0, 0), (0, 100), color=(255, 255, 0, 255), thickness=30
                        )
                dpg.add_separator()
                dpg.add_text(f"Frame: {dpg.get_frame_count()}", tag="frame_text")
                dpg.add_text(f"Time:  {simulation_time}", tag="time_text")

    dpg.create_viewport(
        title="Neuron Simulation", width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT
    )
    dpg.set_viewport_vsync(True)
    dpg.setup_dearpygui()
    update_thread = threading.Thread(
        target=update_series_data, args=(data,), daemon=True
    )
    frame_thread = threading.Thread(target=update_text, daemon=True)
    update_thread.start()
    frame_thread.start()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
