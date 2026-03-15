import pyglet
import numpy as np
from pyglet.shapes import Circle, Rectangle, Line
from pyglet.window import key


class SNNVisualizer:
    def __init__(self, snn, width=2560, height=1440):
        self.window = pyglet.window.Window(
            width, height, caption="SNN", fullscreen=True
        )
        self.batch = pyglet.graphics.Batch()
        self.ui_batch = pyglet.graphics.Batch()
        self.snn = snn

        self.sidebar_x = int(width * 0.86)
        self.mid_y = height // 2
        self.monitored_idx = snn.input_size  # Start monitoring first output neuron

        self.setup_ui()
        self.setup_panes()
        self.setup_network()
        self.window.push_handlers(self)

    def setup_ui(self):
        # Pane Backgrounds and Borders
        self.bg_video = Rectangle(
            self.sidebar_x,
            self.mid_y,
            self.window.width - self.sidebar_x,
            self.mid_y,
            color=(20, 20, 20),
            batch=self.ui_batch,
        )
        self.bg_diag = Rectangle(
            self.sidebar_x,
            0,
            self.window.width - self.sidebar_x,
            self.mid_y,
            color=(20, 20, 20),
            batch=self.ui_batch,
        )

        self.lbl_net = pyglet.text.Label(
            "NETWORK VIZ",
            x=20,
            y=self.window.height - 30,
            font_size=12,
            batch=self.ui_batch,
        )
        self.lbl_vid = pyglet.text.Label(
            "DATA INPUT",
            x=self.sidebar_x + 20,
            y=self.window.height - 30,
            font_size=10,
            batch=self.ui_batch,
        )
        self.lbl_diag = pyglet.text.Label(
            "DIAGNOSTICS",
            x=self.sidebar_x + 20,
            y=self.mid_y - 30,
            font_size=10,
            batch=self.ui_batch,
        )

    def setup_network(self):

        self.neurons = []
        # Layer 1: Input (Standing Vertical Plane)
        for i in range(self.snn.input_size):
            cx, cy = i % 64, i // 64
            nx = 100 + (cx * 5)
            ny = 500 + (cy * 5)
            self.neurons.append(Circle(nx, ny, 2, color=(64, 64, 64), batch=self.batch))

        # Layer 2: Output ch1
        for i in range(self.snn.output_size):
            cx, cy = i % 16, i // 16
            nx = 750 + (cx * 10)
            ny = 500 + (cy * 10)
            self.neurons.append(
                Circle(nx, ny, 4, color=(60, 100, 255), batch=self.batch)
            )
        # Layer 2: Output ch2
        for i in range(self.snn.output_size):
            cx, cy = i % 13, i // 13
            nx = 750 + (cx * 8)
            ny = 800 + (cy * 8)
            self.neurons.append(
                Circle(nx, ny, 4, color=(60, 100, 255), batch=self.batch)
            )
        # Layer 2: Output ch3
        for i in range(self.snn.output_size):
            cx, cy = i % 13, i // 13
            nx = 750 + (cx * 8)
            ny = 900 + (cy * 8)
            self.neurons.append(
                Circle(nx, ny, 4, color=(60, 100, 255), batch=self.batch)
            )

    def setup_panes(self):

        empty_data = np.zeros((64, 64, 3), dtype=np.uint8).tobytes()
        initial_img = pyglet.image.ImageData(64, 64, "RGB", empty_data)
        self.video_sprite = pyglet.sprite.Sprite(
            initial_img, x=self.sidebar_x + 50, y=self.mid_y + 50
        )
        self.video_sprite.scale = 4.0  # 64 * 4 = 256px wide

        self.pot_bar_bg = Rectangle(
            self.sidebar_x + 180, 100, 40, 300, color=(50, 50, 50), batch=self.batch
        )
        self.pot_bar = Rectangle(
            self.sidebar_x + 180, 100, 40, 0, color=(0, 255, 100), batch=self.batch
        )
        self.pot_val_label = pyglet.text.Label(
            "V: 0.000", x=self.sidebar_x + 160, y=70, font_size=10, batch=self.batch
        )
        self.idx_label = pyglet.text.Label(
            f"ID: {self.monitored_idx}",
            x=self.sidebar_x + 60,
            y=self.mid_y - 80,
            font_size=10,
            batch=self.batch,
        )

    def update_frame(self, frame, spikes):
        # Fresh RGB texture update
        rgb = np.dstack((frame, frame, frame))
        self.video_sprite.image = pyglet.image.ImageData(64, 64, "RGB", rgb.tobytes())

        for n in self.neurons:
            n.color = (
                max(40, n.color[0] - 15),
                max(40, n.color[1] - 15),
                max(40, n.color[2] - 15),
            )
        for idx in spikes:
            if idx < len(self.neurons):
                self.neurons[idx].color = (255, 255, 255)

        v = self.snn.v[self.monitored_idx]
        self.pot_bar.height = min(300, v * 300)
        self.pot_val_label = f"V: {v:.4f}"

    def on_draw(self):
        self.window.clear()
        pyglet.gl.glTexParameteri(
            pyglet.gl.GL_TEXTURE_2D,
            pyglet.gl.GL_TEXTURE_MAG_FILTER,
            pyglet.gl.GL_NEAREST,
        )
        self.ui_batch.draw()
        self.batch.draw()
        self.video_sprite.draw()
