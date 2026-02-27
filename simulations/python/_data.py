import numpy as np
from PIL import Image, ImageDraw
import random

MNIST_train_images_path = "data/train-images.idx3-ubyte"
MNIST_test_images_path = "data/t10k-images.idx3-ubyte"


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


def intensity_to_delay_linear(image, T_max=100, T_min=0):
    normalized_image = image.astype(float) / 255.0
    spike_times = T_max - (T_max - T_min) * normalized_image
    return spike_times


def intensity_to_delay_log(image, T_max=100, T_min=0):
    normalized_image = image.astype(float) / 255.0
    spike_times = T_max - (T_max - T_min) * normalized_image
    return spike_times


def intensity_to_inverse_delay_linear(image, T_max=100, T_min=0):
    negative_image = 255 - image
    return intensity_to_delay_linear(negative_image, T_max=T_max, T_min=T_min)


def intensity_to_inverse_delay_log(image, T_max=100, T_min=0):
    negative_image = 255 - image
    return intensity_to_delay_log(negative_image, T_max=T_max, T_min=T_min)


def luminance_to_vector_2d(luminance_image: np.ndarray) -> np.ndarray:
    luminance_image = luminance_image.flatten()
    angles = (luminance_image / 255.0) * 2 * np.pi
    vectors = np.zeros(luminance_image.shape + (2,), dtype=np.float32)
    vectors[..., 0] = np.cos(angles)  # x component
    vectors[..., 1] = np.sin(angles)  # y component
    return vectors


def average_images(image_list):
    stacked_images = np.stack(image_list, axis=0).astype(np.float32)
    averaged_image = np.mean(stacked_images, axis=0)
    return averaged_image.astype(np.uint8)


def parse_MNIST(filename):
    """Read uncompressed MNIST .idx files."""
    with open(filename, "rb") as f:
        magic, size = int.from_bytes(f.read(4), "big"), int.from_bytes(f.read(4), "big")
        if magic == 2049:  # Labels file
            return np.frombuffer(f.read(), dtype=np.uint8)
        elif magic == 2051:  # Images file
            rows, cols = (
                int.from_bytes(f.read(4), "big"),
                int.from_bytes(f.read(4), "big"),
            )
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
        else:
            raise ValueError(f"Unknown magic number {magic} in file {filename}")


def generate_checkerboard(size=32, block_size=4):
    image = np.zeros((size, size), dtype=np.uint8)
    num_blocks = size // block_size
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (i + j) % 2 == 0:
                image[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = np.random.randint(215, 255)
            else:
                image[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = np.random.randint(0, 40)
    return image


def generate_spatiotemporal_stimuli(
    sequence_length,
    grid_size,
    pattern,
    num_injections,
    background_spike_prob=0.01,
    T_max=100,
):
    height, width = grid_size
    stimuli_sequence = []
    for _ in range(sequence_length):
        frame = np.full(grid_size, np.nan)
        random_mask = np.random.rand(height, width) < background_spike_prob
        num_noise_spikes = np.sum(random_mask)
        frame[random_mask] = np.random.uniform(0, T_max, size=num_noise_spikes)
        stimuli_sequence.append(frame)
    p_height, p_width = pattern.shape
    y_start = np.random.randint(0, height - p_height)
    x_start = np.random.randint(0, width - p_width)
    for _ in range(num_injections):
        t_inject = np.random.randint(0, sequence_length)
        injection_slice = stimuli_sequence[t_inject][
            y_start : y_start + p_height, x_start : x_start + p_width
        ]
        combined_spikes = np.fmin(injection_slice, pattern)
        stimuli_sequence[t_inject][
            y_start : y_start + p_height, x_start : x_start + p_width
        ] = combined_spikes
    return stimuli_sequence


def create_random_image_three_channel(width: int, height: int):
    random_pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return random_pixels


def create_random_image_one_channel(width: int, height: int):
    random_pixels = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return random_pixels


def to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    rgb_weights = np.array([0.299, 0.587, 0.114])
    grayscale_image = np.dot(rgb_image[..., :3], rgb_weights)
    return grayscale_image.astype(np.uint8)


def three_to_one_channel(gray_3_channel: np.ndarray) -> np.ndarray:
    if gray_3_channel.ndim != 3 or gray_3_channel.shape[2] != 3:
        raise ValueError(
            "Input must be a 3-channel image with shape (height, width, 3)."
        )
    return gray_3_channel[:, :, 0]


def one_to_three_channel(gray_1_channel: np.ndarray) -> np.ndarray:
    if gray_1_channel.ndim != 2:
        raise ValueError("Input must be a 1-channel image with shape (height, width).")
    return np.stack([gray_1_channel] * 3, axis=-1)


def scale_image(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def rgba_image(image: np.ndarray):
    h, w = image.shape
    rgba_image = np.zeros((h, w, 4), dtype=np.float32)
    rgba_image[..., :3] = image[..., np.newaxis]
    rgba_image[..., 3] = 1.0
    return rgba_image


def inject_pattern(image, rows, cols, value):
    image[rows, cols] = value
