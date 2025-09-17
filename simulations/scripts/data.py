import numpy as np

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

def generate_spatiotemporal_stimuli(sequence_length, grid_size, pattern, num_injections, background_spike_prob=0.01, T_max=100):
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
        injection_slice = stimuli_sequence[t_inject][y_start:y_start+p_height, x_start:x_start+p_width]
        combined_spikes = np.fmin(injection_slice, pattern)
        stimuli_sequence[t_inject][y_start:y_start+p_height, x_start:x_start+p_width] = combined_spikes
    return stimuli_sequence
