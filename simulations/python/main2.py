import pyglet
from pyglet import shapes
import numpy as np
from snn import (
    SpikingNet,
    INPUT_WIDTH,
    INPUT_HEIGHT,
    NUM_FILTERS,
    MAP_WIDTH,
    MAP_HEIGHT,
)
from videogen import LineVideoGenerator

# --- CONFIG ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
NEURON_RADIUS = 3
GRID_SCALE = 12  # Spacing between neurons

# --- SETUP ---
window = pyglet.window.Window(
    WINDOW_WIDTH, WINDOW_HEIGHT, caption="Neuromorphic Efficiency Demo"
)
batch = pyglet.graphics.Batch()

# 1. Initialize World and Brain
gen = LineVideoGenerator(INPUT_WIDTH, INPUT_HEIGHT)
snn = SpikingNet()
snn.load_cnn_weights("cnn_weights.pth")  # Make sure this file exists!

# 2. Pre-calculate Neuron Coordinates for Visualization
neuron_shapes = []
neuron_coords = {}

# Offset for drawing
START_X = 50
START_Y = WINDOW_HEIGHT - 50

# A. Draw Input Layer (28x28 Grid)
for y in range(INPUT_HEIGHT):
    for x in range(INPUT_WIDTH):
        idx = (y * INPUT_WIDTH) + x
        screen_x = START_X + (x * GRID_SCALE)
        screen_y = START_Y - (y * GRID_SCALE)

        circle = shapes.Circle(
            x=screen_x,
            y=screen_y,
            radius=NEURON_RADIUS,
            color=(50, 50, 50),
            batch=batch,
        )
        neuron_shapes.append(circle)
        neuron_coords[idx] = (screen_x, screen_y)

# B. Draw Hidden Layers (4 grids of 13x13)
HIDDEN_START_X = START_X + (INPUT_WIDTH * GRID_SCALE) + 50
for f in range(NUM_FILTERS):
    # Offset each filter block
    filter_offset_x = (f % 2) * (MAP_WIDTH * GRID_SCALE + 20)
    filter_offset_y = (f // 2) * (MAP_HEIGHT * GRID_SCALE + 20)

    start_idx = (INPUT_WIDTH * INPUT_HEIGHT) + (f * MAP_WIDTH * MAP_HEIGHT)

    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            idx = start_idx + (y * MAP_WIDTH) + x
            screen_x = HIDDEN_START_X + filter_offset_x + (x * GRID_SCALE)
            screen_y = START_Y - filter_offset_y - (y * GRID_SCALE)

            circle = shapes.Circle(
                x=screen_x,
                y=screen_y,
                radius=NEURON_RADIUS,
                color=(30, 30, 80),
                batch=batch,
            )  # Dark Blue
            neuron_shapes.append(circle)
            neuron_coords[idx] = (screen_x, screen_y)

# --- LABELS ---
cnn_ops_baseline = 14872  # From our previous calculation
label_stats = pyglet.text.Label("Stats:", x=20, y=100, font_size=14, batch=batch)
label_eff = pyglet.text.Label("Efficiency:", x=20, y=70, font_size=14, batch=batch)
label_frame = pyglet.text.Label("Frame: 0", x=20, y=40, font_size=14, batch=batch)

frame_count = 0
accumulated_snn_ops = 0


def update(dt):
    global frame_count, accumulated_snn_ops

    # 1. Get Video Frame
    frame, (tx, ty) = gen.get_next_frame()

    # 2. Feed to SNN
    snn.set_input_currents(frame)

    # 3. Step Physics (Run 5 steps per frame to allow propagation)
    total_spikes_this_frame = []
    ops_this_frame = 0

    for _ in range(5):
        spikes, ops = snn.advance(dt=1.0)
        total_spikes_this_frame.extend(spikes)
        ops_this_frame += ops

    accumulated_snn_ops += ops_this_frame
    frame_count += 1

    # 4. Update Visuals
    # Reset colors
    for shape in neuron_shapes:
        shape.color = (50, 50, 50)  # Reset to dark gray

    # Highlight Spikes
    for idx in total_spikes_this_frame:
        if idx < len(neuron_shapes):
            # Flash White for spikes
            neuron_shapes[idx].color = (255, 255, 255)

    # 5. Update Stats
    avg_ops = accumulated_snn_ops / frame_count
    ratio = cnn_ops_baseline / (avg_ops + 1)  # Avoid div/0

    label_stats.text = (
        f"SNN Ops (Dynamic): {int(avg_ops)} vs CNN Ops (Fixed): {cnn_ops_baseline}"
    )
    label_eff.text = f"Efficiency Gain: {ratio:.1f}x"
    label_frame.text = f"Frame: {frame_count} | Line Pos: {int(tx)}, {int(ty)}"


@window.event
def on_draw():
    window.clear()
    batch.draw()


# Run
pyglet.clock.schedule_interval(update, 1 / 30.0)  # 30 FPS
pyglet.app.run()
