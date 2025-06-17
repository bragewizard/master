const std = @import("std");
const img = @import("zigimg"); // Assuming this is your image library

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load and process the image
    var file = try std.fs.cwd().openFile("lenna.png", .{});
    defer file.close();

    var image = try img.Image.fromFile(allocator, &file);
    defer image.deinit();
    _ = try image.convert(.grayscale8);

    const bytes = image.rawBytes();
    const width = image.width;
    const height = image.height;

    // Decimation (reduces image size by half in each dimension)
    var new_bytes: [65536]u8 = undefined;
    decimation(bytes, width, &new_bytes);
    const new_width = width / 2;
    const new_height = height / 2;

    // Create negative image for off-center ganglion cells
    var neg_bytes: [65536]u8 = undefined;
    for (new_bytes, 0..) |intensity, i| {
        neg_bytes[i] = 255 - intensity; // Invert intensity (black -> white, white -> black)
    }

    // Process both original (on-center) and negative (off-center) images
    // const PixelInfo = struct { intensity: u8, index: u32 };
    // var pixel_info_on = try allocator.alloc(PixelInfo, new_width * new_height);
    // var pixel_info_off = try allocator.alloc(PixelInfo, new_width * new_height);
    // defer allocator.free(pixel_info_on);
    // defer allocator.free(pixel_info_off);

    // Populate pixel_info for both images
    // for (new_bytes, 0..) |intensity, i| {
    //     pixel_info_on[i] = .{ .intensity = intensity, .index = @intCast(i) };
    //     pixel_info_off[i] = .{ .intensity = neg_bytes[i], .index = @intCast(i) };
    // }

    // Sort pixels by intensity (descending, so higher intensity = earlier firing)
    // const sortFn = struct {
    //     fn lessThan(_: void, a: PixelInfo, b: PixelInfo) bool {
    //         return a.intensity > b.intensity; // Descending order
    //     }
    // }.lessThan;
    // std.sort.sort(PixelInfo, pixel_info_on, {}, sortFn);
    // std.sort.sort(PixelInfo, pixel_info_off, {}, sortFn);

    // Simulate SNN event processing for both on-center and off-center
    // std.debug.print("On-center ganglion cells (original image):\n", .{});
    // for (pixel_info_on, 0..) |info, t| {
    //     const index = info.index;
    //     const x = index % new_width;
    //     const y = index / new_width;
    //     const intensity = info.intensity;
    //     std.debug.print("Time {}: On-center neuron at ({}, {}) fired with intensity {}\n", .{ t, x, y, intensity });
    // }

    // std.debug.print("\nOff-center ganglion cells (negative image):\n", .{});
    // for (pixel_info_off, 0..) |info, t| {
    //     const index = info.index;
    //     const x = index % new_width;
    //     const y = index / new_width;
    //     const intensity = info.intensity;
    //     std.debug.print("Time {}: Off-center neuron at ({}, {}) fired with intensity {}\n", .{ t, x, y, intensity });
    // }

    // Save the decimated and negative images
    var new_img = try img.Image.fromRawPixels(allocator, new_width, new_height, &new_bytes, .grayscale8);
    defer new_img.deinit();
    try new_img.writeToFilePath("lennagray.png", .{ .png = .{} });

    var neg_img = try img.Image.fromRawPixels(allocator, new_width, new_height, &neg_bytes, .grayscale8);
    defer neg_img.deinit();
    try neg_img.writeToFilePath("lennagray_neg.png", .{ .png = .{} });
}
fn convolutionH(image: []const u8, new_bytes: []u8) void {
    const kernel: @Vector(7, u32) = .{ 6, 61, 242, 383, 242, 61, 6 };
    for (0..(image.len - 7)) |i| {
        const vec: @Vector(7, u16) = image[i..][0..7].*;
        const conv = vec * kernel;
        const res = @reduce(.Add, conv);
        new_bytes[i] = @intCast(res >> 10);
    }
}

fn convolutionV(image: img.Image, width: usize, new_bytes: []u8) void {
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
        if (i % 2 == 0 and (i / width) % 2 == 0) {
            new_bytes[index] = pixel;
            index += 1;
        }
    }
}
