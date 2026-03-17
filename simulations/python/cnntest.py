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
    WINDOW_SIZE, WINDOW_SIZE, caption="Donut vs T-Shape Tracker"
)
video_gen = ShapeVideoGenerator(GEN_RES, GEN_RES)
video_batch = pyglet.graphics.Batch()
grid_batch = pyglet.graphics.Batch()

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.BCELoss()

# Sprite setup
empty = np.zeros((GEN_RES, GEN_RES, 3), dtype=np.uint8).tobytes()
sprite = pyglet.sprite.Sprite(
    pyglet.image.ImageData(GEN_RES, GEN_RES, "RGB", empty), batch=video_batch
)
sprite.scale = WINDOW_SIZE / GEN_RES

# 8x8 Grid Indicators
# Red Circle for Donut, Green Square for T-Shape
donut_inds = [
    pyglet.shapes.Circle(0, 0, CELL_SIZE // 3, color=(255, 50, 50), batch=grid_batch)
    for _ in range(64)
]
tshape_inds = [
    pyglet.shapes.Rectangle(
        0, 0, CELL_SIZE - 10, CELL_SIZE - 10, color=(50, 255, 50), batch=grid_batch
    )
    for _ in range(64)
]

for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        # Center the circle, offset the rectangle
        donut_inds[idx].x = j * CELL_SIZE + CELL_SIZE // 2
        donut_inds[idx].y = i * CELL_SIZE + CELL_SIZE // 2
        tshape_inds[idx].x = j * CELL_SIZE + 5
        tshape_inds[idx].y = i * CELL_SIZE + 5


def update(dt):
    frame, coords = video_gen.get_next_frame()
    dx, dy, tx, ty = coords

    target = torch.zeros((1, 2, 8, 8))

    def map_coord(val):
        return int(np.clip((val - 4) / 3.2, 0, 7))

    target[0, 0, map_coord(dy), map_coord(dx)] = 1.0  # Donut
    target[0, 1, map_coord(ty), map_coord(tx)] = 1.0  # T-Shape

    input_tensor = torch.tensor(frame, dtype=torch.float32).view(1, 1, 32, 32) / 255.0

    # 3. Training Step
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)  # Output is [1, 2, 8, 8]
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # 4. Viz Update
    rgb = np.dstack((frame, frame, frame))
    sprite.image = pyglet.image.ImageData(32, 32, "RGB", rgb.tobytes())

    preds = output.detach().numpy()[0]
    for idx in range(64):
        r, c = idx // 8, idx % 8
        # Use a small threshold (0.2) to keep the background clean
        d_val = preds[0, r, c]
        t_val = preds[1, r, c]

        donut_inds[idx].opacity = int(d_val * 255) if d_val > 0.2 else 0
        tshape_inds[idx].opacity = int(t_val * 255) if t_val > 0.2 else 0


@window.event
def on_draw():
    window.clear()
    pyglet.gl.glTexParameteri(
        pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST
    )
    video_batch.draw()
    grid_batch.draw()


pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
