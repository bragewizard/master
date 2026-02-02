import numpy as np
from PIL import Image, ImageDraw
import random


class LineVideoGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.t = 0

        # Line State
        self.x_pos = width // 2
        self.y_pos = height // 2
        self.length = 8  # Short line segment (about 1/3 of the 28px screen)
        self.is_erratic = False

    def get_next_frame(self):
        # 1. Create dark background
        img = Image.new("L", (self.width, self.height), color=0)
        draw = ImageDraw.Draw(img)

        # 2. Update Physics
        self.t += 1

        if self.is_erratic:
            # Teleport randomly
            if random.random() < 0.1:
                self.x_pos = random.randint(2, self.width - 2)
                self.y_pos = random.randint(
                    self.length // 2, self.height - self.length // 2
                )
        else:
            # Smooth Lissajous movement (different frequencies for X and Y)
            # This makes it bounce around the screen like a DVD screensaver

            # Margins so it doesn't hit the absolute edge
            w_amp = (self.width * 0.8) / 2
            h_amp = (self.height * 0.8) / 2

            self.x_pos = (self.width / 2) + np.sin(self.t * 0.05) * w_amp
            # Use Cosine and different speed (0.07) for Y to decouple axes
            self.y_pos = (self.height / 2) + np.cos(self.t * 0.07) * h_amp

        # 3. Draw Line Segment (Vertical) centered at (x, y)
        x = int(self.x_pos)
        y = int(self.y_pos)

        # Calculate start and end Y coordinates based on length
        y_start = max(0, y - self.length // 2)
        y_end = min(self.height, y + self.length // 2)

        draw.line([(x, y_start), (x, y_end)], fill=255, width=3)

        # 4. Return Numpy array and tuple of coordinates
        return np.array(img), (x, y)
