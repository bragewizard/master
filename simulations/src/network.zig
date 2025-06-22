const std = @import("std");

// =============================================================================
//  CONFIGURATION
// =============================================================================

const SimConfig = struct {
    num_neurons: u32 = 50,
    // Simulation should match the generated spike train
    time_steps: u32 = 1000,
    // File containing the input spikes from the previous step
    spike_input_file: []const u8 = "data/spikelist.txt",
    // File to save the final network graph for visualization
    initial_network_output_file: []const u8 = "data/initialnetwork.dot",
    final_network_output_file: []const u8 = "data/finalnetwork.dot",

    // --- Neuron Parameters (Leaky Integrate-and-Fire Model) ---
    potential_decay: f32 = 0.95, // Leak factor per time step (e.g., 5% leak)
    potential_rest: f32 = 0.0,
    potential_threshold: f32 = 1.0,
    potential_reset: f32 = 0.0,
    input_spike_potential: f32 = 0.25, // How much an external input spike increases potential

    // --- Synapse & Learning Parameters (STDP) ---
    initial_connections_per_neuron: u32 = 5,
    // How "far" a neuron can connect initially
    connection_radius: i32 = 5,
    // Learning rates for potentiation (strengthening) and depression (weakening)
    stdp_lr_positive: f32 = 0.005,
    stdp_lr_negative: f32 = 0.003,
    // Time windows for STDP. A smaller tau means spikes must be closer together.
    stdp_tau_positive: f32 = 20.0,
    stdp_tau_negative: f32 = 20.0,
    // Synapse weights will be clamped within this range
    max_weight: f32 = 1.0,
    min_weight: f32 = 0.01,
};
// =============================================================================
//  DATA STRUCTURES
// =============================================================================

const Synapse = struct {
    target_neuron_idx: u32,
    weight: f32,
};

const Neuron = struct {
    // Current state
    potential: f32,
    last_spike_time: u32,

    // List of outgoing connections
    connections: std.ArrayList(Synapse),
};

const Network = struct {
    allocator: std.mem.Allocator,
    neurons: []Neuron,

    pub fn deinit(self: *Network) void {
        for (self.neurons) |*neuron| {
            neuron.connections.deinit();
        }
        self.allocator.free(self.neurons);
    }
};

// =============================================================================
//  NETWORK INITIALIZATION
// =============================================================================

/// Creates and initializes the SNN with random, spatially-close connections.
fn initNetwork(allocator: std.mem.Allocator, config: SimConfig) !Network {
    const neurons = try allocator.alloc(Neuron, config.num_neurons);
    errdefer allocator.free(neurons);

    var prng = std.Random.DefaultPrng.init(1337); // Seed for reproducibility
    const random = prng.random();

    for (0..config.num_neurons) |i| {
        neurons[i] = Neuron{
            .potential = config.potential_rest,
            .last_spike_time = 0, // Initialize to 0
            .connections = std.ArrayList(Synapse).init(allocator),
        };

        // Create initial outgoing connections
        var k: u32 = 0;
        while (k < config.initial_connections_per_neuron) {
            // Pick a target neuron `j` that is spatially close to `i`
            const offset: i32 = random.intRangeAtMost(i32, -config.connection_radius, config.connection_radius);
            var j: i32 = @as(i32, @intCast(i)) + offset;

            // Handle wrap-around for neurons at the edges
            if (j < 0) j += @intCast(config.num_neurons);
            if (j >= config.num_neurons) j -= @intCast(config.num_neurons);

            // Don't connect a neuron to itself
            if (j == i) continue;

            try neurons[i].connections.append(Synapse{
                .target_neuron_idx = @intCast(j),
                .weight = random.float(f32) * 0.1, // Start with small random weights
            });
            k += 1;
        }
    }

    return Network{
        .allocator = allocator,
        .neurons = neurons,
    };
}

// =============================================================================
//  SPIKE DATA LOADER
// =============================================================================

/// Loads spike data from the specified file into a format that's fast to access during simulation.
/// Returns an ArrayList where the index is the time step, and the value is a slice of neuron indices that spiked.
fn loadSpikeInput(allocator: std.mem.Allocator, config: SimConfig) !std.ArrayList([]u32) {
    var spike_map = std.ArrayList([]u32).init(allocator);
    errdefer spike_map.deinit();
    try spike_map.resize(config.time_steps);

    // Initialize slices
    for (0..config.time_steps) |t| {
        spike_map.items[t] = &.{};
    }

    // Temporary storage for spikes at each time step
    var temp_spikes = std.ArrayList(std.ArrayList(u32)).init(allocator);
    errdefer {
        for (temp_spikes.items) |list| list.deinit();
        temp_spikes.deinit();
    }
    try temp_spikes.resize(config.time_steps);
    for (0..config.time_steps) |t| {
        temp_spikes.items[t] = std.ArrayList(u32).init(allocator);
    }

    // Read the file and populate temp_spikes
    const file = try std.fs.cwd().openFile(config.spike_input_file, .{});
    defer file.close();

    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();
    var line_buf: [1024]u8 = undefined;

    while (try in_stream.readUntilDelimiterOrEof(&line_buf, '\n')) |line| {
        if (line.len == 0 or line[0] == '#') continue;

        var it = std.mem.splitScalar(u8, line, ' ');
        const neuron_str = it.next() orelse continue;
        const time_str = it.next() orelse continue;

        const neuron_idx = try std.fmt.parseInt(u32, neuron_str, 10);
        const time_step = try std.fmt.parseInt(u32, time_str, 10);

        if (time_step < config.time_steps) {
            try temp_spikes.items[time_step].append(neuron_idx);
        }
    }

    // Convert temp ArrayLists to final slices
    for (0..config.time_steps) |t| {
        spike_map.items[t] = try temp_spikes.items[t].toOwnedSlice();
    }

    // Deallocate temporary structure
    for (temp_spikes.items) |list| list.deinit();
    temp_spikes.deinit();

    return spike_map;
}

// =============================================================================
//  NETWORK VISUALIZATION EXPORT
// =============================================================================

/// Saves the network structure to a .dot file for visualization with Graphviz.
fn saveNetworkAsDot(network: Network, filename: []const u8, weight_threshold: f32) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();
    var writer = file.writer();

    try writer.writeAll("digraph SNN {\n");
    try writer.writeAll("    rankdir=\"LR\";\n");
    try writer.writeAll("    bgcolor=\"#222222\";\n");
    try writer.writeAll("    node [shape=circle, style=filled, fillcolor=\"#44aadd\", fontcolor=white, color=white];\n");
    try writer.writeAll("    edge [color=\"#cccccc\"];\n\n");

    // Define all nodes first
    for (0..network.neurons.len) |i| {
        try writer.print("    {d};\n", .{i});
    }
    try writer.writeAll("\n");

    // Define all edges with weights as attributes
    for (0..network.neurons.len) |i| {
        for (network.neurons[i].connections.items) |synapse| {
            // Only draw connections that meet the weight threshold
            if (synapse.weight > weight_threshold) {
                // penwidth makes the line thicker for stronger weights
                const pen_width = 1.0 + (synapse.weight * 4.0);
                // opacity makes the line more solid for stronger weights
                const opacity = 0.2 + (synapse.weight * 0.8);

                try writer.print("    {d} -> {d} [penwidth={d:.2}, color=\"#00ffff{X:0>2}\", tooltip=\"{d:.4}\"];\n", .{
                    i,
                    synapse.target_neuron_idx,
                    pen_width,
                    @as(u8, @intFromFloat(opacity * 255)), // Append opacity as hex
                    synapse.weight,
                });
            }
        }
    }
    try writer.writeAll("}\n");
}

// =============================================================================
//  SIMULATION
// =============================================================================
pub fn run(allocator: std.mem.Allocator) !void {
    const config = SimConfig{};

    // --- SETUP ---
    std.debug.print("Initializing network...\n", .{});
    var network = try initNetwork(allocator, config);
    defer network.deinit();

    // *** ADDED: SAVE INITIAL STATE ***
    std.debug.print("Saving initial network graph to '{s}'...\n", .{config.initial_network_output_file});
    try saveNetworkAsDot(network, config.initial_network_output_file, 0.0); // Threshold 0.0 to show ALL connections

    std.debug.print("Loading spike data from '{s}'...\n", .{config.spike_input_file});
    const spike_input = try loadSpikeInput(allocator, config);
    defer {
        for (spike_input.items) |s| allocator.free(s);
        spike_input.deinit();
    }
    std.debug.print("Setup complete. Starting simulation...\n", .{});

    // --- SIMULATION LOOP ---
    for (0..config.time_steps) |t| {
        if (@rem(t, 100) == 0) {
            std.debug.print("  -> Time step {d}/{d}\n", .{ t, config.time_steps });
        }

        // Apply external spikes from the input file
        for (spike_input.items[t]) |neuron_idx| {
            network.neurons[neuron_idx].potential += config.input_spike_potential;
        }

        var fired_neurons = std.ArrayList(u32).init(allocator);
        defer fired_neurons.deinit();

        // Update each neuron's potential and check for firing
        for (0..config.num_neurons) |i| {
            var n = &network.neurons[i];
            // 1. Decay potential (leak)
            n.potential *= config.potential_decay;

            // 2. Check for firing
            if (n.potential > config.potential_threshold) {
                try fired_neurons.append(@intCast(i)); // Record that this neuron fired
                n.potential = config.potential_reset; // Reset potential
                n.last_spike_time = @intCast(t); // Record time of spike for STDP

                // Propagate potential to target neurons
                for (n.connections.items) |synapse| {
                    network.neurons[synapse.target_neuron_idx].potential += synapse.weight;
                }
            }
        }

        // --- NEW STDP LOGIC (Corrected and Symmetrical) ---
        // For every neuron that just fired, update both its incoming and outgoing synapses.
        for (fired_neurons.items) |fired_idx| {
            // 1. Potentiation (LTP): Update INCOMING synapses.
            //    Find all neurons `pre_idx` that connect to `fired_idx`.
            for (0..config.num_neurons) |pre_idx| {
                if (pre_idx == fired_idx) continue;
                for (network.neurons[pre_idx].connections.items) |*synapse| {
                    if (synapse.target_neuron_idx == fired_idx) {
                        const dt: f32 = @as(f32, @floatFromInt(t)) - @as(f32, @floatFromInt(network.neurons[pre_idx].last_spike_time));
                        if (dt > 0) { // pre-before-post causes potentiation
                            const dw = config.stdp_lr_positive * std.math.exp(-dt / config.stdp_tau_positive);
                            synapse.weight += dw;
                            synapse.weight = std.math.clamp(synapse.weight, config.min_weight, config.max_weight);
                        }
                        break;
                    }
                }
            }

            // 2. Depression (LTD): Update OUTGOING synapses.
            //    For each connection going out from `fired_idx`.
            for (network.neurons[fired_idx].connections.items) |*synapse| {
                const post_idx = synapse.target_neuron_idx;
                // `fired_idx` fired at `t`, `post_idx` fired at `last_spike_time`.
                const dt: f32 = @as(f32, @floatFromInt(t)) - @as(f32, @floatFromInt(network.neurons[post_idx].last_spike_time));
                if (dt > 0) { // post-before-pre causes depression
                    const dw = config.stdp_lr_negative * std.math.exp(-dt / config.stdp_tau_negative);
                    synapse.weight -= dw;
                    synapse.weight = std.math.clamp(synapse.weight, config.min_weight, config.max_weight);
                }
            }
        }
    }
    std.debug.print("Simulation finished.\n\n", .{});

    // --- REPORT & SAVE FINAL RESULTS ---
    std.debug.print("Saving final network graph to '{s}'...\n", .{config.final_network_output_file});
    try saveNetworkAsDot(network, config.final_network_output_file, 0.1); // Threshold 0.1 to show only strong connections
    std.debug.print("Done. Use a tool like Graphviz to render the .dot files.\n", .{});
    const stdout = std.io.getStdOut().writer();
    try stdout.print("# Final Synaptic Weights\n", .{});
    try stdout.print("# Format: <from_neuron> -> <to_neuron> [weight]\n", .{});

    for (0..config.num_neurons) |i| {
        for (network.neurons[i].connections.items) |synapse| {
            // Only print connections that have strengthened significantly
            if (synapse.weight > 0.1) {
                try stdout.print("{d} -> {d} [{d:.4}]\n", .{ i, synapse.target_neuron_idx, synapse.weight });
            }
        }
    }
}
