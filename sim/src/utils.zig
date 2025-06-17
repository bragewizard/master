pub fn argsort(comptime T: type, allocator: std.mem.Allocator, values: []const T) ![]usize {
    const len = values.len;

    // Step 1: Create an array of indices
    var indices = try allocator.alloc(usize, len);
    for (indices) |_, i| {
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
