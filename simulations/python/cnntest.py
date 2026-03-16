import pyglet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from _data import ShapeVideoGenerator
from _cnn import SimpleCNN

# --- CONFIG ---
GEN_RES = 32
WINDOW_SIZE = 800
GRID_SIZE = 8
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

# --- SETUP ---
window = pyglet.window.Window(
    WINDOW_SIZE, WINDOW_SIZE, caption="CNN Grid Trainer (32x32)"
)
video_gen = ShapeVideoGenerator(GEN_RES, GEN_RES)
video_batch = pyglet.graphics.Batch()
grid_batch = pyglet.graphics.Batch()

# 1. Model & Training
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.BCELoss()  # Binary Cross Entropy for Grid Cells

# 2. Video Sprite (32x32 scaled to 800x800)
empty = np.zeros((GEN_RES, GEN_RES, 3), dtype=np.uint8).tobytes()
sprite = pyglet.sprite.Sprite(
    pyglet.image.ImageData(GEN_RES, GEN_RES, "RGB", empty), batch=video_batch
)
sprite.scale = WINDOW_SIZE / GEN_RES

# 3. Grid Visualizer
# Red = Square Detection, Green = Triangle Detection
square_indicators = []
triangle_indicators = []

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        # Square (Red)
        s = pyglet.shapes.Rectangle(
            j * CELL_SIZE,
            i * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
            color=(255, 0, 0),
            batch=grid_batch,
        )
        s.opacity = 0
        square_indicators.append(s)
        # Triangle (Green)
        t = pyglet.shapes.Rectangle(
            j * CELL_SIZE + 10,
            i * CELL_SIZE + 10,
            CELL_SIZE - 20,
            CELL_SIZE - 20,
            color=(0, 255, 0),
            batch=grid_batch,
        )
        t.opacity = 0
        triangle_indicators.append(t)


def get_grid_labels(gen):
    """Converts generator coordinates to an 8x8 target tensor."""
    target = torch.zeros((1, 2, GRID_SIZE, GRID_SIZE))

    # Map 32px space to 8 cells (32/8 = 4px per cell)
    sq_gx = int(np.clip(gen.x_sq / 4, 0, 7))
    sq_gy = int(np.clip(gen.y_sq / 4, 0, 7))

    tri_gx = int(np.clip(gen.x_tri / 4, 0, 7))
    tri_gy = int(np.clip(gen.y_tri / 4, 0, 7))

    target[0, 0, sq_gy, sq_gx] = 1.0  # Channel 0: Square
    target[0, 1, tri_gy, tri_gx] = 1.0  # Channel 1: Triangle
    return target


def update(dt):
    # --- 1. GET DATA ---
    raw_frame, _ = video_gen.get_next_frame()
    target = get_grid_labels(video_gen)

    # Input tensor [1, 1, 32, 32]
    input_tensor = (
        torch.tensor(raw_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    )

    # --- 2. TRAIN STEP ---
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)  # (1, 2, 8, 8)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # --- 3. UI UPDATE ---
    # Refresh Video
    rgb_frame = np.dstack((raw_frame, raw_frame, raw_frame))
    sprite.image = pyglet.image.ImageData(GEN_RES, GEN_RES, "RGB", rgb_frame.tobytes())

    # Update Visual Detections
    preds = output.detach().numpy()[0]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            # Map probability [0, 1] to opacity [0, 255]
            square_indicators[idx].opacity = int(preds[0, i, j] * 255)
            triangle_indicators[idx].opacity = int(preds[1, i, j] * 255)


@window.event
def on_draw():
    window.clear()

    # Force OpenGL to use Nearest Neighbor (No blur/interpolation)
    pyglet.gl.glTexParameteri(
        pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST
    )
    pyglet.gl.glTexParameteri(
        pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MIN_FILTER, pyglet.gl.GL_NEAREST
    )

    video_batch.draw()
    grid_batch.draw()


pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
