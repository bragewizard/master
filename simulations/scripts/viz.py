import matplotlib.pyplot as plt
import numpy as np
import sys

def visualize_spikes(file_path):
    """Reads spike data from a file and creates a raster plot."""
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

    # plt.style.use('dark_background')
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["FreeMono"]
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_spike_data.txt>")
    else:
        visualize_spikes(sys.argv[1])
