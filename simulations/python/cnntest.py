import pyglet
from pyglet import shapes
import numpy as np
import torch
from _cnn import SimpleLineCNN, FRAME_HEIGHT, FRAME_WIDTH
from _data import LineVideoGenerator
import os

# --- CONFIGURATION ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Visual Config (Dot Matrix)
GRID_SPACING = 20  # Pixels between dots
DOT_RADIUS = 5
GRID_START_X = (WINDOW_WIDTH - (FRAME_WIDTH * GRID_SPACING)) // 2
GRID_START_Y = (
    WINDOW_HEIGHT + (FRAME_HEIGHT * GRID_SPACING)
) // 2 - 50  # Start from top

WEIGHTS_FILE = "cnn_weights.pth"


# --- SETUP ---
window = pyglet.window.Window(
    WINDOW_WIDTH, WINDOW_HEIGHT, caption="Experiment 3: CNN Inference (Dot Matrix View)"
)
batch = pyglet.graphics.Batch()

# Groups to ensure order (Background dots first, Indicators on top)
background_group = pyglet.graphics.Group(order=0)
foreground_group = pyglet.graphics.Group(order=1)

# 1. World & Brain
gen = LineVideoGenerator(FRAME_WIDTH, FRAME_HEIGHT)
model = SimpleLineCNN()

# 2. Load Weights
if os.path.exists(WEIGHTS_FILE):
    try:
        model.load_state_dict(torch.load(WEIGHTS_FILE))
        model.eval()
        print(f"SUCCESS: Loaded {WEIGHTS_FILE}")
    except Exception as e:
        print(f"ERROR: Architecture mismatch. {e}")
else:
    print(f"WARNING: {WEIGHTS_FILE} not found. Using random weights.")

# 3. Create the Dot Matrix (Retina)
# We store the circle objects in a list so we can update their color later
pixel_dots = []

for y in range(FRAME_HEIGHT):
    for x in range(FRAME_WIDTH):
        # Calculate screen position
        # Note: Grid Y goes Top->Down, Pyglet Y goes Bottom->Up
        screen_x = GRID_START_X + (x * GRID_SPACING)
        screen_y = GRID_START_Y - (y * GRID_SPACING)

        dot = shapes.Circle(
            x=screen_x,
            y=screen_y,
            radius=DOT_RADIUS,
            color=(50, 50, 50),  # Start Dark Gray
            batch=batch,
            group=background_group,
        )
        pixel_dots.append(dot)

# 4. Create Indicators (Red/Green Dots)
# We make them larger and translucent so they don't fully obscure the pixels
target_dot = shapes.Circle(
    x=0,
    y=0,
    radius=DOT_RADIUS * 2.5,
    color=(0, 255, 0, 150),  # Green, Semi-transparent
    batch=batch,
    group=foreground_group,
)

pred_dot = shapes.Circle(
    x=0,
    y=0,
    radius=DOT_RADIUS * 2.0,
    color=(255, 50, 50, 180),  # Red, Semi-transparent
    batch=batch,
    group=foreground_group,
)

# Labels
label_info = pyglet.text.Label(
    "CNN TRACKER", font_size=16, x=GRID_START_X, y=WINDOW_HEIGHT - 40, batch=batch
)
label_legend = pyglet.text.Label(
    "GREEN: Ground Truth | RED: CNN Prediction",
    font_size=12,
    x=GRID_START_X,
    y=WINDOW_HEIGHT - 65,
    batch=batch,
)
label_coords = pyglet.text.Label("", font_size=12, x=GRID_START_X, y=50, batch=batch)


def logical_to_screen(lx, ly):
    """
    Converts logical 28x28 coordinates (0,0 top-left)
    to screen pixel coordinates (based on our grid).
    """
    sx = GRID_START_X + (lx * GRID_SPACING)
    sy = GRID_START_Y - (ly * GRID_SPACING)
    return sx, sy


def update(dt):
    # 1. Get Data
    raw_frame, (true_x, true_y) = gen.get_next_frame()

    # 2. Run Inference
    input_tensor = torch.tensor(
        raw_frame[np.newaxis, np.newaxis, :, :] / 255.0, dtype=torch.float32
    )
    with torch.no_grad():
        prediction = model(input_tensor)

    # 3. Update Retina (The Grid)
    # Flatten frame to match our flat list of dots
    flat_pixels = raw_frame.flatten()

    for i, intensity in enumerate(flat_pixels):
        # Update color: Black (0) -> Dark Gray (50), White (255) -> White (255)
        val = int(intensity)
        if val > 50:
            pixel_dots[i].color = (val, val, val)
        else:
            pixel_dots[i].color = (30, 30, 30)  # Background dim color

    # 4. Update Indicators

    # Prediction: [-1, 1] -> [0, 28]
    pred_x_norm = prediction[0][0].item()
    pred_y_norm = prediction[0][1].item()

    pred_x = ((pred_x_norm + 1) / 2) * FRAME_WIDTH
    pred_y = ((pred_y_norm + 1) / 2) * FRAME_HEIGHT

    # Move the dots
    t_sx, t_sy = logical_to_screen(true_x, true_y)
    p_sx, p_sy = logical_to_screen(pred_x, pred_y)

    target_dot.x, target_dot.y = t_sx, t_sy
    pred_dot.x, pred_dot.y = p_sx, p_sy

    label_coords.text = (
        f"Target: ({int(true_x)}, {int(true_y)}) | Pred: ({int(pred_x)}, {int(pred_y)})"
    )


@window.event
def on_draw():
    window.clear()
    batch.draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.E:
        gen.is_erratic = not gen.is_erratic
        print(f"Erratic Mode: {gen.is_erratic}")


# Run
pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
