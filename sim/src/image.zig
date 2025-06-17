const std = @import("std");
const img = @import("zigimg");
const Allocator = std.mem.Allocator;

/// Convert a png image to an 3D array of values representing
/// the intensity at each pixel in the R G and B channels
pub fn pngToArray(allocator: Allocator, path: []const u8) ![]f32 {

    const image = try img.Image.fromFilePath(allocator, path);
    return image;

    // const width = image.width;
    // const height = image.height;
    // const channels = switch (image.pixels) {
    //     img.PixelFormat.rgba32 => 4,
    //     img.PixelFormat.rgb24 => 3,
    //     else => return error.UnsupportedFormat,
    // };

    // const tensor_size = width * height * channels;
    // var tensor = try allocator.alloc(f32, tensor_size);

    // for (image.data[0..tensor_size], 0..) |pixel, i| {
    //     tensor[i] = @as(f32, @floatFromInt(pixel)) / 255.0;
    // }

    // return tensor;
}

test "png to array" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            std.log.err("LEAK :(", .{});
        } else {
            std.log.info("you good homie :)", .{});
        }
    }

    const allocator = gpa.allocator();

    var image = try pngToArray(allocator, "../lenna.png");
    defer image.deinit();
}
