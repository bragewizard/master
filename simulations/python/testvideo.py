import pyglet
import numpy as np
from _data import LineVideoGenerator, ShapeVideoGenerator

# --- CONFIG ---
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
GEN_WIDTH = 32
GEN_HEIGHT = 32

# --- SETUP ---
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Video Feed Test")
# video_gen = LineVideoGenerator(GEN_WIDTH, GEN_HEIGHT)
video_gen = ShapeVideoGenerator(GEN_WIDTH, GEN_HEIGHT)
batch = pyglet.graphics.Batch()

# Create a placeholder sprite (we will replace its image every frame)
# Initial dummy image
empty_data = np.zeros((GEN_HEIGHT, GEN_WIDTH, 3), dtype=np.uint8).tobytes()
initial_image = pyglet.image.ImageData(GEN_WIDTH, GEN_HEIGHT, "RGB", empty_data)
sprite = pyglet.sprite.Sprite(initial_image, batch=batch)

# Scale up so it fills the window
scale_x = WINDOW_WIDTH / GEN_WIDTH
scale_y = WINDOW_HEIGHT / GEN_HEIGHT
sprite.scale_x = scale_x
sprite.scale_y = scale_y

label = pyglet.text.Label(
    "Pos: 0", font_size=24, x=10, y=WINDOW_HEIGHT - 30, batch=batch
)


def update(dt):
    # 1. Get new frame (Shape: 28x28)
    raw_frame, true_x = video_gen.get_next_frame()

    # DEBUG: Uncomment this to prove the generator is working
    # if np.max(raw_frame) == 0:
    #     print("FRAME IS PURE BLACK! Generator issue.")
    # else:
    #     print(f"Frame max val: {np.max(raw_frame)}")

    # 2. Convert to RGB (Shape: 28x28x3)
    rgb_frame = np.dstack((raw_frame, raw_frame, raw_frame))

    # 3. Create NEW Image Data
    # This forces Pyglet to treat it as a fresh texture, ensuring the update is visible
    new_image = pyglet.image.ImageData(
        GEN_WIDTH, GEN_HEIGHT, "RGB", rgb_frame.tobytes()
    )

    # 4. Update Sprite
    sprite.image = new_image

    # Update label
    label.text = f"Target X: {true_x}"


@window.event
def on_draw():
    window.clear()

    # Optional: Draw a gray background so you can see the black video frame edge
    # (Pyglet clear usually does black, so this isn't strictly necessary if video is black)

    # Ensure "Pixelated" look (Nearest Neighbor)
    pyglet.gl.glTexParameteri(
        pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST
    )

    batch.draw()


# Run at 60 FPS
pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
