const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorLength = cpu_tensor.tensorLength;
const tensorStride = cpu_tensor.tensorStride;
const expectEqual = @import("../testing.zig").expectEqual;

fn index(tensor: var, cartesian_index: []const usize) @TypeOf(tensor).ScalarType {
    var linear_index: usize = 0;
    for (tensor.stride) |s, i| {
        assert(cartesian_index[i] < tensor.shape[i]);
        linear_index += s * cartesian_index[i];
    }
    return tensor.storage.array[linear_index];
}

test "cpu tensor index" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    std.testing.expectEqual(index(tensor, &[_]usize{ 0, 0 }), 1);
    std.testing.expectEqual(index(tensor, &[_]usize{ 0, 1 }), 2);
    std.testing.expectEqual(index(tensor, &[_]usize{ 0, 2 }), 3);
    std.testing.expectEqual(index(tensor, &[_]usize{ 1, 0 }), 4);
    std.testing.expectEqual(index(tensor, &[_]usize{ 1, 1 }), 5);
    std.testing.expectEqual(index(tensor, &[_]usize{ 1, 2 }), 6);
}

pub fn matrix_multiply(allocator: *Allocator, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    if (x.shape[1] != y.shape[0])
        return error.ShapeMismatch;
    const m = x.shape[0];
    const n = x.shape[1];
    const p = y.shape[1];
    const T = @TypeOf(x);
    const shape = try allocator.alloc(usize, 2);
    errdefer allocator.free(shape);
    shape[0] = m;
    shape[1] = p;
    const stride = try tensorStride(2, allocator, shape);
    const length = tensorLength(shape);
    const new_array = try allocator.alloc(T.ScalarType, length);
    errdefer allocator.free(new_array);
    const tensor = T{
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = new_array },
    };
    var i: usize = 0;
    while (i < m - 1) : (i += 1) {
        var j: usize = 0;
        while (j < p - 1) : (j += 1) {
            var k: usize = 0;
            while (k < n - 1) : (k += 1) {
                index(tensor, &[_]usize{ i, j }) += index(x, &[_]usize{ i, k }) * index(y, &[_]usize{ k, j });
            }
        }
    }
    return tensor;
}

test "matrix_multiply" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(&arena.allocator, [_][3]f16{
        .{ 1, -2, 3 },
        .{ -4, 5, -6 },
    });
    const actual = try matrix_multiply(&arena.allocator, x, y);
    const expected = try constant(&arena.allocator, [_][3]f16{
        .{ 1, 4, 1 },
        .{ 9, 16, 2 },
        .{ 25, 36, 4 },
    });
    // expectEqual(actual, expected);
}
