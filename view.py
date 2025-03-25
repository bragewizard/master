import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def read_idx(filename):
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

# Paths to your unzipped MNIST files
train_images_path = "train-images.idx3-ubyte"
test_images_path = "t10k-images.idx3-ubyte"

# Load the datasets
train_images = read_idx(train_images_path)
test_images = read_idx(test_images_path)

# Normalize image data to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Print dataset shapes
print(f"Train Images: {train_images.shape}")
print(f"Test Images: {test_images.shape}")


def chunk_image(image, chunk_size):
    """Divide the image into non-overlapping chunks of size (chunk_size x chunk_size)."""
    h, w = image.shape
    assert h % chunk_size == 0 and w % chunk_size == 0, "Image dimensions must be divisible by chunk_size."
    chunks = []
    for i in range(0, h, chunk_size):
        for j in range(0, w, chunk_size):
            chunk = image[i:i + chunk_size, j:j + chunk_size]
            chunks.append(chunk)
    return chunks


def visualize_chunks(chunks):
    """Visualize the image chunks."""
    n = len(chunks)
    size = int(np.sqrt(n))  # Assuming chunks form a square grid
    fig, axes = plt.subplots(size, size, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(chunks[i], cmap='gray')
        ax.axis('off')
    plt.show()


# Function to convert image chunks to spike times
def image_to_spikes_absolute(chunks, max_delay=100):
    """Convert chunks into spikes with absolute temporal coding."""
    spikes = []
    for chunk in chunks:
        # flat_chunk = chunk.flatten()
        spike_times = max_delay - (chunk / 1.0) * max_delay  # Absolute delay
        # spike_times.reshape(4,4)
        spikes.append(spike_times)
        # spike_times = spike_times[spike_times < 100]
    return np.array(spikes)


# Function to visualize spikes in 3D
def visualize_3d_spikes_with_image(spikes, chunk_size, image, image_shape):
    """Create a 3D raster plot for spikes, with the original image as background in the 3D grid."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a meshgrid for the image to align it with the spike grid in 3D
    x_image, y_image = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    
    # Flatten the meshgrid to align with 3D space
    x_image = x_image.flatten()
    y_image = y_image.flatten()
    z_image = np.zeros_like(x_image)  # Set the z-coordinates of the image in 3D
    
    # Plot the image as a background (2D surface in 3D space)
    ax.scatter(z_image, x_image, np.flip(y_image), c=image.flatten(), cmap='gray', s=10)  # Image surface
    
    # Initialize lists to hold the data for the 3D plot
    times = []  # Spike times
    neurons = []  # Neuron IDs (2D coordinates of the chunk)
    chunks = []  # Chunk index
    
    # # For each chunk, add spike data to the lists
    for i,chunk in enumerate(spikes):
        chunk_row = i // (image_shape[1] // chunk_size)
        chunk_col = i % (image_shape[1] // chunk_size)    #     # Determine the (x, y) position of each chunk in the image space
        
        for neuron_id, spike_time in enumerate(chunk.flatten()):
            if (spike_time < 100):
                # Calculate neuron position in the original image coordinates
                x_pos = chunk_col * chunk_size + (neuron_id % chunk_size)
                y_pos = chunk_row * chunk_size + (neuron_id // chunk_size)
            
                # Add data for plotting
                times.append(spike_time)
                neurons.append((x_pos, y_pos))  # Use (x, y) coordinates
    
    # # Convert lists to numpy arrays
    times = np.array(times).flatten()
    neurons = np.array(neurons)
    chunks = np.array(chunks)
    
    # # Unzip neurons (x, y) coordinates
    x_neurons, y_neurons = neurons[:, 0], neurons[:, 1]
    
    # # Plot the 3D scatter plot for spikes
    ax.scatter(times,x_neurons,np.flip(y_neurons), marker='*', color='b', s=10)  # Spike events
    
    # Labels and title
    ax.set_xlabel('Time (Spike Delay)')
    ax.set_ylabel('X Position (Pixel)')
    ax.set_zlabel('Y Position (Pixel)')
    ax.set_title('3D Spike Raster Plot with Image Overlay')

    # Show plot
    plt.show()
# Example MNIST image (assuming you have loaded train_images)

example_image = train_images[2]  # Select the first training image
plt.imshow(example_image, cmap='gray')
# Step 3: Visualize the raster plot for the first chunk

# Step 1: Divide the image into chunks
chunk_size = 4  # Each chunk is 4x4 pixels
chunks = chunk_image(example_image, chunk_size)
visualize_chunks(chunks)

# Step 2: Convert chunks into spikes
spikes = image_to_spikes_absolute(chunks)
print(spikes.shape)

# Step 3: Visualize the 3D raster plot of spikes
visualize_3d_spikes_with_image(spikes, chunk_size,example_image,example_image.shape)
# Example MNIST image (assuming you have loaded train_images)




# Initialize parameters
m = 16  # Number of input neurons (pixels in a 4x4 chunk)
n = 4  # Number of output neurons
learning_rate = 0.1  # STDP learning rate

# Initialize weights randomly (binary weights: 0 or 1)
weights = np.random.choice([0, 1], size=(m, n))

def simulate_layer(input_spike_times, weights):
    """Simulate a single layer of spiking neurons."""
    # Initialize membrane potentials
    membrane_potentials = np.zeros(weights.shape[1])
    spike_outputs = np.zeros(weights.shape[1])  # Output spikes (binary)
    
    # Process input spikes
    for neuron_id, spike_time in enumerate(input_spike_times):
        if spike_time < np.inf:  # Ignore non-spiking neurons
            # Add weighted input to membrane potentials
            membrane_potentials += weights[neuron_id]
    
    # Generate output spikes
    spike_outputs[membrane_potentials >= n] = 1
    
    return spike_outputs, membrane_potentials

def stdp_update(input_spike_times, output_spikes, weights, learning_rate):
    """Apply STDP learning rule."""
    for input_id, spike_time in enumerate(input_spike_times):
        for output_id, spike in enumerate(output_spikes):
            if spike:  # If the output neuron fired
                # Strengthen weights for spiking input neurons
                weights[input_id, output_id] += learning_rate
                # Clip weights to binary (0 or 1)
                weights[input_id, output_id] = min(1, weights[input_id, output_id])
    return weights
