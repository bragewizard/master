const std = @import("std");

pub fn readFile(allocator: std.mem.Allocator, filename: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    const size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, size);
    _ = try file.read(buffer);
    return buffer;
}

pub fn argsort(comptime T: type, allocator: std.mem.Allocator, values: []const T) ![]usize {
    const len = values.len;

    // Step 1: Create an array of indices
    var indices = try allocator.alloc(usize, len);
    for (indices) |i| {
        indices[i] = i;
    }

    // Step 2: Sort the indices based on values
    std.sort.sort(usize, indices, values, struct {
        pub fn lessThan(context: []const T, a: usize, b: usize) bool {
            return context[a] < context[b];
        }
    }.lessThan);

    return indices;
}

// TODO: optimize and use vector if it is faster (need to benchmark)
pub fn convoluteH(image: []const u8, new_bytes: []u8) void {
    const kernel: @Vector(7, u32) = .{ 6, 61, 242, 383, 242, 61, 6 };
    for (0..(image.len - 7)) |i| {
        const vec: @Vector(7, u16) = image[i..][0..7].*;
        const conv = vec * kernel;
        const res = @reduce(.Add, conv);
        new_bytes[i] = @intCast(res >> 10);
    }
}

pub fn convoluteV(image: []const u8, width: usize, new_bytes: []u8) void {
    const kernel: @Vector(7, u32) = .{ 6, 61, 242, 383, 242, 61, 6 };
    for (0..(image.len - 7)) |i| {
        const vec: @Vector(7, u16) = .{
            image[0],
            image[width],
            image[width * 1],
            image[width * 2],
            image[width * 3],
            image[width * 4],
            image[width * 5],
        };
        const conv = vec * kernel;
        const res = @reduce(.Add, conv);
        new_bytes[i] = @intCast(res >> 10);
    }
}

pub fn decimate(image: []const u8, width: usize, new_bytes: []u8) void {
    var index: u32 = 0;
    for (image, 0..) |pixel, i| {
        if (i % 2 == 0 and (i / width) % 2 == 0) {
            new_bytes[index] = pixel;
            index += 1;
        }
    }
}
