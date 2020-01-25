const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorLength = cpu_tensor.tensorLength;
const tensorStride = cpu_tensor.tensorStride;
const expectEqual = @import("../testing.zig").expectEqual;

fn linear_index(stride: []const usize, cartesian_index: []const usize) usize {
    var index: usize = 0;
    for (stride) |s, i| index += s * cartesian_index[i];
    return index;
}

test "linear index" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const stride = tensor.stride;
    std.testing.expectEqual(linear_index(stride, &[_]usize{ 0, 0 }), 0);
    std.testing.expectEqual(linear_index(stride, &[_]usize{ 0, 1 }), 1);
    std.testing.expectEqual(linear_index(stride, &[_]usize{ 0, 2 }), 2);
    std.testing.expectEqual(linear_index(stride, &[_]usize{ 1, 0 }), 3);
    std.testing.expectEqual(linear_index(stride, &[_]usize{ 1, 1 }), 4);
    std.testing.expectEqual(linear_index(stride, &[_]usize{ 1, 2 }), 5);
}

pub fn matrix_multiply(allocator: *Allocator, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    if (x.shape[1] != y.shape[0])
        return error.ShapeMismatch;
    const x_array = x.storage.array;
    const y_array = y.storage.array;
    const m = x.shape[0];
    const n = x.shape[1];
    const p = y.shape[1];
    const T = @TypeOf(x);
    const shape = try allocator.alloc(usize, 2);
    errdefer allocator.free(shape);
    shape[0] = m;
    shape[1] = p;
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    const length = tensorLength(shape);
    const tensor_array = try allocator.alloc(T.ScalarType, length);
    errdefer allocator.free(new_array);
    const tensor = T{
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = tensor_array },
    };
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < p) : (j += 1) {
            const t_index = linear_index(tensor.stride, &[_]usize{ i, j });
            tensor_array[t_index] = 0;
            var k: usize = 0;
            while (k < n) : (k += 1) {
                const x_index = linear_index(x.stride, &[_]usize{ i, k });
                const y_index = linear_index(y.stride, &[_]usize{ k, j });
                tensor_array[t_index] += x_array[x_index] * y_array[y_index];
            }
        }
    }
    return tensor;
}

test "matrix_multiply" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]i32{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ 5, -6 },
    });
    const y = try constant(&arena.allocator, [_][3]i32{
        .{ 1, -2, 3 },
        .{ -4, 5, -6 },
    });
    const actual = try matrix_multiply(&arena.allocator, x, y);
    const expected = try constant(&arena.allocator, [_][3]i32{
        .{ 9, -12, 15 },
        .{ 19, -26, 33 },
        .{ 29, -40, 51 },
    });
    expectEqual(actual, expected);
}
