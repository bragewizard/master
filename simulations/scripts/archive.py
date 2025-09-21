def calculate_first_layer_conv(image, on_filter, off_filter, T_max=100, T_min=0):
    height, width = image.shape
    output_height = height - 2
    output_width = width - 2
    on_spike_times = np.full((output_height, output_width), np.nan)
    off_spike_times = np.full((output_height, output_width), np.nan)
    normalized_image = image.astype(float) / 255.0
    max_potential = on_filter[1, 1]
    for y in range(output_height):
        for x in range(output_width):
            patch = normalized_image[y:y+3, x:x+3]
            on_potential = np.sum(patch * on_filter)
            off_potential = np.sum(patch * off_filter)
            if on_potential > 0:
                scaled_potential = min(on_potential / max_potential, 1.0)
                on_spike_times[y, x] = T_max - (T_max - T_min) * scaled_potential
            if off_potential > 0:
                scaled_potential = min(off_potential / max_potential, 1.0)
                off_spike_times[y, x] = T_max - (T_max - T_min) * scaled_potential
    return on_spike_times, off_spike_times

def calculate_first_layer_bi_encoding(image, filter_pos, filter_neg, threshold=3):
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
    return firing_times

def location_wise_wta(list_of_spike_time_grids):
    if not list_of_spike_time_grids:
        return np.array([])
    combined_min_times = list_of_spike_time_grids[0]
    for i in range(1, len(list_of_spike_time_grids)):
        combined_min_times = np.fmin(combined_min_times, list_of_spike_time_grids[i])
    return combined_min_times


def create_on_off_filters(center_weight=8, surround_weight=-1):
    on_filter = np.full((3, 3), surround_weight, dtype=float)
    on_filter[1, 1] = center_weight
    off_filter = np.full((3, 3), -surround_weight, dtype=float) # Note the sign flip
    off_filter[1, 1] = -center_weight
    return on_filter, off_filter

def calculate_first_layer_conv(image, on_filter, off_filter, T_max=100, T_min=0):
    height, width = image.shape
    output_height = height - 2
    output_width = width - 2
    on_spike_times = np.full((output_height, output_width), np.nan)
    off_spike_times = np.full((output_height, output_width), np.nan)
    normalized_image = image.astype(float) / 255.0
    max_potential = on_filter[1, 1]
    for y in range(output_height):
        for x in range(output_width):
            patch = normalized_image[y:y+3, x:x+3]
            on_potential = np.sum(patch * on_filter)
            off_potential = np.sum(patch * off_filter)
            if on_potential > 0:
                scaled_potential = min(on_potential / max_potential, 1.0)
                on_spike_times[y, x] = T_max - (T_max - T_min) * scaled_potential
            if off_potential > 0:
                scaled_potential = min(off_potential / max_potential, 1.0)
                off_spike_times[y, x] = T_max - (T_max - T_min) * scaled_potential
    return on_spike_times, off_spike_times
