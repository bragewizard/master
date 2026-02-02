import numpy as np

MNIST_train_images_path = "data/train-images.idx3-ubyte"
MNIST_test_images_path = "data/t10k-images.idx3-ubyte"


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
