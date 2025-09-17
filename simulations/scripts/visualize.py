import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import numpy as np

def visualizeSpikes(file_path):
    spikes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    neuron_idx = int(parts[0])
                    time_step = int(parts[1])
                    spikes.append((time_step, neuron_idx))
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return

    if not spikes:
        print("No spike data found to visualize.")
        return

    times, neurons = zip(*spikes)
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["GeistMono NF"]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.scatter(times, neurons, marker='|', s=20, color='black')

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Spike Train")
    ax.set_ylim(min(neurons) - 1, max(neurons) + 1)
    ax.set_xlim(0, max(times) + 1)
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


def visualizeStimuliAnimation(stimuli_sequence, title="Spatiotemporal Stimuli"):
    if not stimuli_sequence:
        print("Stimuli sequence is empty.")
        return

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    
    # Use viridis_r so earlier spikes (lower numbers) are brighter
    img = ax.imshow(stimuli_sequence[0], cmap='viridis_r', vmin=0, vmax=100)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Spike Time (ms)')
    
    def update(frame_index):
        img.set_data(stimuli_sequence[frame_index])
        ax.set_title(f"{title} (Frame {frame_index})")
        return [img]

    ani = FuncAnimation(fig, update, frames=len(stimuli_sequence), blit=True, interval=100)
    plt.show()

def plotNeuronVoltageTrace(title, time, V, Ie, Ii):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(time, V, label='Voltage (V)', color='black')
    ax1.axhline(15, ls='--', color='red', label='Threshold')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    ax2.plot(time, Ie, label='Excitatory Current (I_exc)', color='green')
    ax2.plot(time, Ii, label='Inhibitory Current (I_inh)', color='orange')
    ax2.plot(time, Ie + Ii, label='Total Current', color='blue', ls=':')
    ax2.set_ylabel('Synaptic Current')
    ax2.set_xlabel('Time (ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    plt.show()

def visualizeStaticRasterPlot(spike_times, title="Static Raster Plot"):
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

def visualizeSequenceRasterPlot(stimuli_sequence, frame_duration=100, title="Spatiotemporal Raster Plot"):
    if not stimuli_sequence:
        print("Stimuli sequence is empty.")
        return
    grid_size = stimuli_sequence[0].shape
    height, width = grid_size
    num_neurons = height * width
    all_spike_times = []
    all_neuron_indices = []
    for t, frame in enumerate(stimuli_sequence):
        y_coords, x_coords = np.where(~np.isnan(frame))
        if y_coords.size > 0:
            times_in_frame = frame[y_coords, x_coords]
            global_times = t * frame_duration + times_in_frame
            neuron_indices = y_coords * width + x_coords
            all_spike_times.extend(global_times.tolist())
            all_neuron_indices.extend(neuron_indices.tolist())

    plt.figure(figsize=(12, 7))
    plt.scatter(all_spike_times, all_neuron_indices, s=15, c='black', marker='|')
    
    plt.title(title)
    plt.xlabel("Global Time (ms)")
    plt.ylabel("Neuron Index")
    
    total_duration = len(stimuli_sequence) * frame_duration
    plt.xlim(0, total_duration)
    plt.ylim(0, num_neurons)
    
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.show()

def visualizeImage(image, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualizeSpikeTimes(spike_times, title="Spike Times"):
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

def visualizeRasterPlot(spike_times, title="Raster Plot"):
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

def visualize3dSpikes(spike_times, title="3D Spiking Plot"):
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
