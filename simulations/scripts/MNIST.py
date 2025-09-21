import numpy as np

MNIST_train_images_path = "data/train-images.idx3-ubyte"
MNIST_test_images_path = "data/t10k-images.idx3-ubyte"

def parse_MNIST(filename):
    """Read uncompressed MNIST .idx files."""
    with open(filename, 'rb') as f:
        magic, size = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
        if magic == 2049:  # Labels file
            return np.frombuffer(f.read(), dtype=np.uint8)
        elif magic == 2051:  # Images file
            rows, cols = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
        else:
            raise ValueError(f"Unknown magic number {magic} in file {filename}")
