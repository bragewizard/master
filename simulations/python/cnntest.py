import pyglet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from _data import ShapeVideoGenerator
from _cnn import SimpleCNN

# --- CONFIG ---
GEN_WIDTH, GEN_HEIGHT = 64, 64
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 800

# --- SETUP ---
window = pyglet.window.Window(
    WINDOW_WIDTH, WINDOW_HEIGHT, caption="CNN Tracking Diagnostic"
)
video_gen = ShapeVideoGenerator(GEN_WIDTH, GEN_HEIGHT)
video_batch = pyglet.graphics.Batch()
foreground_batch = pyglet.graphics.Batch()  # Dedicated batch for markers

# 1. Model & Training Setup
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()

# 2. Video Feed Sprite (Background)
empty_data = np.zeros((GEN_HEIGHT, GEN_WIDTH, 3), dtype=np.uint8).tobytes()
initial_image = pyglet.image.ImageData(GEN_WIDTH, GEN_HEIGHT, "RGB", empty_data)
sprite = pyglet.sprite.Sprite(initial_image, batch=video_batch)
sprite.scale = WINDOW_WIDTH / GEN_WIDTH

# 3. VISUAL MARKERS (Foreground)
# CNN PREDICTIONS - Solid Bright Colors
pred_sq = pyglet.shapes.Rectangle(
    0, 0, 30, 30, color=(255, 0, 0), batch=foreground_batch
)
pred_sq.opacity = 180
pred_sq.anchor_x, pred_sq.anchor_y = 15, 15

pred_tri = pyglet.shapes.Circle(0, 0, 15, color=(0, 255, 0), batch=foreground_batch)
pred_tri.opacity = 180

label = pyglet.text.Label(
    "Initializing...", x=10, y=20, font_size=14, batch=foreground_batch
)


def update(dt):
    # --- 1. GET DATA ---
    raw_frame, (sq_tx, sq_ty) = video_gen.get_next_frame()
    tri_tx, tri_ty = int(video_gen.x_tri), int(video_gen.y_tri)

    target = torch.tensor(
        [[sq_tx / 64.0, sq_ty / 64.0, tri_tx / 64.0, tri_ty / 64.0]],
        dtype=torch.float32,
    )
    input_tensor = (
        torch.tensor(raw_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    )

    # --- 2. TRAIN ---
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # --- 3. UPDATE VISUALS ---
    rgb_frame = np.dstack((raw_frame, raw_frame, raw_frame))
    sprite.image = pyglet.image.ImageData(64, 64, "RGB", rgb_frame.tobytes())

    # Update CNN Prediction Markers
    preds = output.detach().numpy()[0]
    # Note: Pyglet Y starts at bottom. If video looks inverted, use: (1.0 - preds[1])
    pred_sq.x, pred_sq.y = preds[0] * WINDOW_WIDTH, preds[1] * WINDOW_HEIGHT
    pred_tri.x, pred_tri.y = preds[2] * WINDOW_WIDTH, preds[3] * WINDOW_HEIGHT

    label.text = f"Loss: {loss.item():.6f} | Red=Square, Green=Triangle"


@window.event
def on_draw():
    window.clear()

    # Order matters:
    # 1. Draw Video Batch first
    video_batch.draw()

    # 2. Draw Foreground Batch (Markers/Labels) second
    foreground_batch.draw()

    # Ensure pixelated scaling
    pyglet.gl.glTexParameteri(
        pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST
    )


pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
