import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

def intensity_to_delay_encoding(image, T_max=100, T_min=0):
    normalized_image = image.astype(float) / 255.0
    spike_times = T_max - (T_max - T_min) * normalized_image
    return spike_times

def negative_image_encoding(image, T_max=100, T_min=0):
    negative_image = 255 - image
    return intensity_to_delay_encoding(negative_image, T_max=T_max, T_min=T_min)

def integrate_and_fire(excitatory, inhibitory, threshold=1.0):
    sorted_indices = np.argsort(excitatory)
    sorted_times = excitatory[sorted_indices]
    integrated_potential = 0.0
    firing_time = None
    for i in range(len(sorted_times)):
        integrated_potential += 1
        
        if integrated_potential >= threshold:
            firing_time = sorted_times[i]
            break
    return firing_time

def visualize_image(image, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_spike_times(spike_times, title="Spike Times"):
    plt.figure(figsize=(6, 6))
    masked_spike_times = np.ma.masked_where(np.isnan(spike_times), spike_times)
    plt.imshow(masked_spike_times, cmap='viridis_r', interpolation='nearest', vmin=0, vmax=100)
    cbar = plt.colorbar(label='Spike Time (ms)')
    cbar.ax.tick_params(labelsize=8)
    plt.title(title)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

def visualize_raster_plot(spike_times, title="Raster Plot"):
    plt.figure(figsize=(10, 6))
    height, width = spike_times.shape
    y_coords, x_coords = np.where(~np.isnan(spike_times))
    times = spike_times[y_coords, x_coords]
    neuron_indices = y_coords * width + x_coords
    plt.scatter(times, neuron_indices, s=10, c='black')
    plt.title(title)
    plt.xlabel("Spike Time (ms)")
    plt.ylabel("Neuron Index (y * width + x)")
    plt.xlim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def visualize_3d_spikes(spike_times, title="3D Spiking Plot"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    y_coords, x_coords = np.where(~np.isnan(spike_times))
    times = spike_times[y_coords, x_coords]
    colors = plt.cm.viridis_r(times / 100)
    ax.scatter(x_coords, y_coords, times, c=colors, s=20)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Spike Time (ms)')
    ax.invert_yaxis()
    plt.show()

def apply_full_filter_grid(image, filter_pos, filter_neg, threshold=3):
    height, width = image.shape
    output_height = height - 2
    output_width = width - 2
    
    firing_times = np.full((output_height, output_width), np.nan)
    pos_spikes = intensity_to_delay_encoding(image)
    neg_spikes = negative_image_encoding(image)
    for y in range(output_height):
        for x in range(output_width):
            pos_patch = pos_spikes[y:y+3, x:x+3]
            neg_patch = neg_spikes[y:y+3, x:x+3]
            synapses = []
            for i in range(3):
                for j in range(3):
                    if filter_pos[i,j] == 1:
                        synapses.append(pos_patch[i, j])
                    if filter_neg[i,j] == 1:
                        synapses.append(neg_patch[i, j])
            firing_time = integrate_and_fire(np.array(synapses), None, threshold)
            if firing_time is not None:
                firing_times[y, x] = firing_time
            # if firing_time > 80:
                # firing_times[y, x] = firing_time
    return firing_times

def generate_checkerboard(size=32, block_size=4):
    image = np.zeros((size, size), dtype=np.uint8)
    num_blocks = size // block_size
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (i + j) % 2 == 0:
                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = np.random.randint(215,255)
            else:
                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = np.random.randint(0,40)
    return image

def location_wise_wta(list_of_spike_time_grids):
    if not list_of_spike_time_grids:
        return np.array([])
    combined_min_times = list_of_spike_time_grids[0]
    for i in range(1, len(list_of_spike_time_grids)):
        combined_min_times = np.fmin(combined_min_times, list_of_spike_time_grids[i])
    return combined_min_times


def run_pipeline(input_image, title):
    print(f"\n--- Processing: {title} ---")
    visualize_image(input_image, f"Input Image: {title}")
    
    filter_pos_v_edge = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    filter_neg_v_edge = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    filter_pos_h_edge = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    filter_neg_h_edge = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    
    filter_pos_diag1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    filter_neg_diag1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    filter_pos_diag2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    filter_neg_diag2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print("Applying all filters...")
    pos_spikes = intensity_to_delay_encoding(input_image)
    neg_spikes = negative_image_encoding(input_image)
    visualize_spike_times(pos_spikes, f"pos")
    visualize_spike_times(neg_spikes, f"neg")
    visualize_raster_plot(pos_spikes, f"pos")
    visualize_raster_plot(neg_spikes, f"neg")
    v_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_v_edge, filter_neg_v_edge, threshold=6)
    v_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_v_edge, filter_pos_v_edge, threshold=6)
    h_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_h_edge, filter_neg_h_edge, threshold=6)
    h_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_h_edge, filter_pos_h_edge, threshold=6)
    d1_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_diag1, filter_neg_diag1, threshold=6)
    d1_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_diag1, filter_pos_diag1, threshold=6)
    d2_pos_edge_times = apply_full_filter_grid(input_image, filter_pos_diag2, filter_neg_diag2, threshold=6)
    d2_neg_edge_times = apply_full_filter_grid(input_image, filter_neg_diag2, filter_pos_diag2, threshold=6)
    
    visualize_spike_times(v_pos_edge_times, f"v_pos_edge")
    visualize_spike_times(h_pos_edge_times, f"h_pos_edge")
    visualize_spike_times(d1_pos_edge_times, f"d1_pos_edge")
    visualize_spike_times(d2_pos_edge_times, f"d2_pos_edge")
    print("Performing location-wise Winner-Take-All...")
    location_wta_output = location_wise_wta([v_pos_edge_times, v_neg_edge_times,
                                             h_pos_edge_times, h_neg_edge_times,
                                             d1_pos_edge_times, d1_neg_edge_times,
                                             d2_pos_edge_times, d2_neg_edge_times])
    visualize_spike_times(location_wta_output, f"edge")
    
if __name__ == "__main__":

    # image_path = 'data/doomguy.jpg'
    image_path = 'data/lenna.png'
    original_image = Image.open(image_path)
    grayscale_image = original_image.convert('L')
    max_dim = 256
    low_res_image_pil = grayscale_image.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    # input_image = np.array(low_res_image_pil)
    input_image = generate_checkerboard(size=256, block_size=32)
    run_pipeline(input_image, "edge detection")
