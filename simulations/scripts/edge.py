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

def apply_filter(positive_spikes, negative_spikes, filter_pos, filter_neg, threshold=1.0):
    times = np.concatenate([positive_spikes.flatten(), negative_spikes.flatten()])
    weights = np.concatenate([filter_pos.flatten(), filter_neg.flatten()])
    sorted_indices = np.argsort(times)
    sorted_times = times[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_sum = 0
    firing_time = None
    for i in range(len(sorted_times)):
        cumulative_sum += sorted_weights[i]
        if cumulative_sum >= threshold:
            firing_time = sorted_times[i]
            break
    return firing_time

def winner_take_all(neuron_firing_times, inhibition_radius=1):
    output_times = np.full_like(neuron_firing_times, np.nan)
    height, width = neuron_firing_times.shape
    
    valid_fire_idx = ~np.isnan(neuron_firing_times)
    fire_times_flat = neuron_firing_times[valid_fire_idx].flatten()
    original_flat_indices = np.where(valid_fire_idx.flatten())[0]

    if fire_times_flat.size == 0:
        return output_times

    sorted_order = np.argsort(fire_times_flat)
    sorted_original_flat_indices = original_flat_indices[sorted_order]
    
    inhibited_mask = np.zeros_like(neuron_firing_times, dtype=bool)

    for flat_idx in sorted_original_flat_indices:
        y, x = np.unravel_index(flat_idx, (height, width))
        
        if inhibited_mask[y, x]:
            continue
        
        for dy in range(-inhibition_radius, inhibition_radius + 1):
            for dx in range(-inhibition_radius, inhibition_radius + 1):
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < height and 0 <= nx < width:
                    inhibited_mask[ny, nx] = True # Mark as inhibited
        output_times[y, x] = neuron_firing_times[y, x]
            
    return output_times

def visualize_image(image, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_spike_times(spike_times, title="Spike Times"):
    plt.figure(figsize=(6, 6))
    masked_spike_times = np.ma.masked_where(np.isnan(spike_times), spike_times)
    plt.imshow(masked_spike_times, cmap='hot_r', interpolation='nearest', vmin=0, vmax=100)
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
            firing_time = apply_filter(pos_patch, neg_patch, filter_pos, filter_neg, threshold)
            if firing_time is not None:
                firing_times[y, x] = firing_time
    return firing_times

def generate_checkerboard(size=32, block_size=4):
    image = np.zeros((size, size), dtype=np.uint8)
    num_blocks = size // block_size
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (i + j) % 2 == 0:
                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 255
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
    
    filter_pos_v_edge = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    filter_neg_v_edge = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    filter_pos_h_edge = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    filter_neg_h_edge = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    
    filter_pos_diag1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    filter_neg_diag1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    filter_pos_diag2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    filter_neg_diag2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print("Applying all filters...")
    v_edge_times = apply_full_filter_grid(input_image, filter_pos_v_edge, filter_neg_v_edge, threshold=3)
    h_edge_times = apply_full_filter_grid(input_image, filter_pos_h_edge, filter_neg_h_edge, threshold=3)
    d1_edge_times = apply_full_filter_grid(input_image, filter_pos_diag1, filter_neg_diag1, threshold=3)
    d2_edge_times = apply_full_filter_grid(input_image, filter_pos_diag2, filter_neg_diag2, threshold=3)
    
    print("Performing location-wise Winner-Take-All...")
    location_wta_output = location_wise_wta([v_edge_times, h_edge_times, d1_edge_times, d2_edge_times])
    
    visualize_spike_times(location_wta_output, f"Location-wise WTA Output (All Edges) - {title}")

    print("Applying spatial Winner-Take-All...")
    final_sparse_output = winner_take_all(location_wta_output, inhibition_radius=1)
    
    visualize_spike_times(final_sparse_output, f"Final Sparse Output (Spatial WTA) - {title}")
    visualize_3d_spikes(final_sparse_output, f"Final 3D Spiking Plot - {title}")
    visualize_raster_plot(final_sparse_output, f"Final Raster Plot - {title}")
    
    
if __name__ == "__main__":

    image_path = 'data/doomguy.jpg'
    original_image = Image.open(image_path)
    grayscale_image = original_image.convert('L')
    max_dim = 64
    low_res_image_pil = grayscale_image.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    input_image = np.array(low_res_image_pil)
    # input_image = generate_checkerboard(size=32, block_size=4)
    run_pipeline(input_image, "edge detection")
