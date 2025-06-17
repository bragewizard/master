//! SpikeLists: A structure for managing lists of spike events in a neural simulation.
//!
//! This implementation maintains an array of `CoordArray` structures, where each `CoordArray`
//! stores a list of spike event coordinates. The system allows initialization, re-initialization,
//! writing to a file, and iterating over spike lists.
//!
//! Features:
//! - **Initialization (`init`)**: Allocates memory for spike lists.
//! - **Re-initialization (`reInit`)**: Resets spike list sizes and updates pointers.
//! - **Writing (`write`)**: Saves spike data to a file in a structured format.
//! - **Display (`affiche`)**: Prints the first spike list's contents to the console.
//! - **Iteration (`nextSpikeList`)**: Moves to the next spike list in a circular buffer fashion.
//! - **Memory Management (`deinit`)**: Ensures proper cleanup of allocated memory.
//!
//! Usage:
//! 1. Initialize `SpikeLists` with a specified number of spike lists.
//! 2. Add spike event data to individual lists.
//! 3. Write spike data to a file.
//! 4. Display or iterate over stored spike events.
//! 5. Properly deallocate resources after use.
//!
//! This is useful in simulations where neuronal spike timing needs to be stored
//! and processed efficiently.

const std = @import("std");

const MAX_SPIKE_LIST = 1000;
const MAX_SPIKE = 500;
const YES = 1;

const Coord = struct {
    x: i32,
    y: i32,
};

const CoordArray = struct {
    array: []Coord,
    size: usize,

    pub fn init(allocator: std.mem.Allocator) !CoordArray {
        return CoordArray{
            .array = try allocator.alloc(Coord, MAX_SPIKE),
            .size = 0,
        };
    }

    pub fn deinit(self: *CoordArray, allocator: std.mem.Allocator) void {
        allocator.free(self.array);
    }
};

const SpikeLists = struct {
    all_spike_lists: []CoordArray,
    nb: usize,
    current_ptr: isize,
    next_list: *CoordArray,

    pub fn init(allocator: std.mem.Allocator, nb_reserv: usize) !SpikeLists {
        if (nb_reserv > MAX_SPIKE_LIST) return error.MaxLatencyOverflow;
        var all_spike_lists = try allocator.alloc(CoordArray, nb_reserv);
        for (0..nb_reserv) |i| {
            all_spike_lists[i] = try CoordArray.init(allocator);
        }
        return SpikeLists{
            .all_spike_lists = all_spike_lists,
            .nb = nb_reserv,
            .current_ptr = -1,
            .next_list = &all_spike_lists[0],
        };
    }

    pub fn reInit(self: *SpikeLists) void {
        self.current_ptr = 0;
        self.next_list = &self.all_spike_lists[1];
        for (0..self.nb) |i| {
            self.all_spike_lists[i].size = 0;
        }
    }

    pub fn write(self: *SpikeLists, nb2write: usize, fileName: []const u8) !void {
        var file = try std.fs.cwd().createFile(fileName, .{});
        defer file.close();

        var writer = file.writer();
        try writer.print("{}\n", .{nb2write});

        for (0..nb2write) |i| {
            const spike_list = &self.all_spike_lists[i];
            try writer.print("{}\t", .{spike_list.size});
            for (0..spike_list.size) |j| {
                try writer.print("{}\t{}\t", .{ spike_list.array[j].x, spike_list.array[j].y });
            }
            try writer.writeAll("\n");
        }
    }

    pub fn affiche(self: *SpikeLists) void {
        const spike_list = &self.all_spike_lists[0];
        std.debug.print("{}\n", .{spike_list.size});
        for (0..spike_list.size) |i| {
            std.debug.print("{}-{}\n", .{ spike_list.array[i].x, spike_list.array[i].y });
        }
    }

    pub fn nextSpikeList(self: *SpikeLists, reinit: i32) void {
        self.current_ptr = (self.current_ptr + 1) % @as(isize, @intCast(self.nb));
        self.next_list = &self.all_spike_lists[@mod(@as(usize, @intCast(self.current_ptr + 1)), self.nb)];
        if (reinit == YES) self.next_list.size = 0;
    }

    pub fn deinit(self: *SpikeLists, allocator: std.mem.Allocator) void {
        for (self.all_spike_lists) |*list| {
            list.deinit(allocator);
        }
        allocator.free(self.all_spike_lists);
    }
};

// Example usage
test "spikelist" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var spike_lists = try SpikeLists.init(allocator, 10);
    defer spike_lists.deinit(allocator);

    try spike_lists.write(5, "spike_output.txt");
    spike_lists.affiche();
}
