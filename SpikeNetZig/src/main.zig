//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.
const img = @import("zigimg");
const std = @import("std");

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var file = try std.fs.cwd().openFile("lenna.png", .{});
    defer file.close();

    var image = try img.Image.fromFile(allocator, &file);
    defer image.deinit();
    _ = try image.convert(.grayscale8);
    const bytes = image.rawBytes();
    std.debug.print("{}\n", .{bytes.len});
    var max:u8 = bytes[0];
    for (bytes) |p| {
        if (p > max) {
            max = p;
        }
    }
    std.debug.print("{}\n", .{max});
    const width = image.width;
    const height = image.height;
    // two buffers
    var new_bytes: [300000]u8 = undefined;
    var new_bytes2: [300000]u8 = undefined;

    convolutionH(bytes, &new_bytes);
    convolutionH(&new_bytes, &new_bytes2);
    convolutionH(&new_bytes2, &new_bytes);
    // decimation(&new_bytes2,image.width, &new_bytes);
    // var new_img = try img.Image.fromRawPixels(allocator, width / 4, height / 4, &new_bytes2, .grayscale8);
    var new_img = try img.Image.fromRawPixels(allocator, width, height, &new_bytes, .grayscale8);
    defer new_img.deinit();
    try new_img.writeToFilePath("lennagray.png", .{ .png = .{} });
    // try image.writeToFilePath("lennagray.png", .{ .png = .{} });
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}

fn convolutionH(image:[]const u8, new_bytes: []u8) void {
    const kernel: @Vector(7, u32) = .{ 6, 61, 242, 383, 242, 61, 6 };
    for (0..(image.len - 7)) |i| {
        const vec: @Vector(7, u16) = image[i..][0..7].*;
        const conv = vec * kernel;
        const res = @reduce(.Add, conv);
        new_bytes[i] = @intCast(res >> 10);
    }
}


fn convolutionV(image: img.Image, width:usize, new_bytes: []u8) void {
    const kernel: @Vector(7, u32) = .{ 6, 61, 242, 383, 242, 61, 6 };
    const bytes = image.rawBytes();
    for (0..(bytes.len - 7)) |i| {
        const vec: @Vector(7, u16) = .{
            bytes[0],
            bytes[width],
            bytes[width * 1],
            bytes[width * 2], 
            bytes[width * 3], 
            bytes[width * 4], 
            bytes[width * 5],
        }; 
        const conv = vec * kernel;
        const res = @reduce(.Add, conv);
        new_bytes[i] = @intCast(res >> 10);
    }
}

fn decimation(image: []const u8, width: usize, new_bytes: []u8) void {
    var index: u32 = 0;
    for (image, 0..) |pixel, i| {
        if (i % 4 == 0 and (i / width) % 4 == 0) {
            new_bytes[index] = pixel;
            index += 1;
        }
    }
}
