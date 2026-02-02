const std = @import("std");

/// Represents the configuration for the spike train generation.
const SpikeTrainConfig = struct {
    /// Total number of neurons in the network.
    num_neurons: u32,
    /// Total duration of the simulation in time steps.
    time_steps: u32,
    /// Probability (0.0 to 1.0) of a single neuron firing randomly at any given time step.
    random_spike_prob: f32,
    /// The specific pattern to insert. It's a slice of relative (time, neuron) coordinates.
    /// Example: .{ .{0, 1}, .{1, 3}, .{2, 5} } means neuron 1 fires at t=0,
    /// neuron 3 at t=1, and neuron 5 at t=2, relative to the pattern's start time.
    pattern: []const struct { time_offset: u32, neuron_idx: u32 },
    /// The pattern will be inserted at regular intervals of this many time steps.
    pattern_interval: u32,
};

/// Main function to generate and print the spike train.
pub fn run(allocator: std.mem.Allocator) !void {
    // --- Configuration ---
    // Here you can define the parameters for your spike train.
    const config = SpikeTrainConfig{
        .num_neurons = 50,
        .time_steps = 1000,
        .random_spike_prob = 0.02, // 2% chance of a random spike per neuron per step

        // A simple, recognizable pattern: a diagonal sweep across 5 neurons.
        .pattern = &.{
            .{ .time_offset = 0, .neuron_idx = 10 },
            .{ .time_offset = 1, .neuron_idx = 11 },
            .{ .time_offset = 2, .neuron_idx = 12 },
            .{ .time_offset = 3, .neuron_idx = 13 },
            .{ .time_offset = 4, .neuron_idx = 14 },
        },
        .pattern_interval = 200, // Insert the pattern every 200 time steps.
    };

    // --- Data Structure ---
    // We'll use a 2D slice to represent the spike train: spikes[time][neuron]
    // A value of 1 means a spike, 0 means no spike.
    var spike_train = try allocator.alloc([]u8, config.time_steps);
    defer {
        for (spike_train) |time_slice| {
            allocator.free(time_slice);
        }
        allocator.free(spike_train);
    }

    for (0..config.time_steps) |t| {
        spike_train[t] = try allocator.alloc(u8, config.num_neurons);
        // Initialize all to 0 (no spike)
        @memset(spike_train[t], 0);
    }

    // --- Spike Generation Logic ---

    // 1. Generate the background random spikes (noise)
    var prng = std.Random.DefaultPrng.init(0); // Using seed 0 for reproducibility
    const random = prng.random();

    for (0..config.time_steps) |t| {
        for (0..config.num_neurons) |n| {
            if (random.float(f32) < config.random_spike_prob) {
                spike_train[t][n] = 1;
            }
        }
    }

    // 2. Insert the recurring pattern over the noise
    var current_time: u32 = 0;
    while (current_time < config.time_steps) {
        // Check if it's time to insert the pattern
        if (@rem(current_time, config.pattern_interval) == 0) {
            // Insert each spike from the pattern definition
            for (config.pattern) |pattern_spike| {
                const pattern_time = current_time + pattern_spike.time_offset;
                const pattern_neuron = pattern_spike.neuron_idx;

                // Ensure we don't write past the end of the simulation
                if (pattern_time < config.time_steps and pattern_neuron < config.num_neurons) {
                    // Overwrite any existing random spike with the pattern spike.
                    spike_train[pattern_time][pattern_neuron] = 1;
                }
            }
        }
        current_time += 1;
    }

    // --- Output the Spike Train ---
    // We print in a simple "neuron_index time_step" format for spikes only.
    // This is more efficient than printing a full matrix for sparse spikes.
    const stdout = std.io.getStdOut().writer();
    try stdout.print("# Spike train data\n", .{});
    try stdout.print("# Format: <neuron_index> <time_step>\n", .{});
    try stdout.print("# Config: {d} neurons, {d} steps, {d:.2} noise prob, pattern interval {d}\n", .{
        config.num_neurons,
        config.time_steps,
        config.random_spike_prob,
        config.pattern_interval,
    });

    for (0..config.time_steps) |t| {
        for (0..config.num_neurons) |n| {
            if (spike_train[t][n] == 1) {
                try stdout.print("{d} {d}\n", .{ n, t });
            }
        }
    }
}
