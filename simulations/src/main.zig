const std = @import("std");
const img = @import("zigimg");
const readFile = @import("utils.zig").readFile;
const mnist = @import("mnist.zig");
const simplePatterns = @import("simplePatterns.zig");
const network = @import("network.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    // try simplePatterns.run(allocator);
    try network.run(allocator);
}
