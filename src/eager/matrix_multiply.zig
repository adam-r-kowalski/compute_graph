const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorLength = cpu_tensor.tensorLength;
const tensorStride = cpu_tensor.tensorStride;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

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

pub fn matrixMultiply(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    if (x.shape[1] != y.shape[0])
        return error.ShapeMismatch;
    const x_array = x.storage.array;
    const y_array = y.storage.array;
    const m = x.shape[0];
    const n = x.shape[1];
    const p = y.shape[1];
    const shape = try allocator.alloc(usize, 2);
    errdefer allocator.free(shape);
    shape[0] = m;
    shape[1] = p;
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    const length = tensorLength(shape);
    const tensor_array = try allocator.alloc(T, length);
    errdefer allocator.free(new_array);
    const tensor = CpuTensor(T){
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

test "matrixMultiply" {
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
    const actual = try matrixMultiply(i32, &arena.allocator, x, y);
    const expected = try constant(&arena.allocator, [_][3]i32{
        .{ 9, -12, 15 },
        .{ 19, -26, 33 },
        .{ 29, -40, 51 },
    });
    expectEqual(i32, actual, expected);
}

fn transpose(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    var shape = try allocator.alloc(usize, 2);
    errdefer allocator.free(shape);
    shape[0] = tensor.shape[1];
    shape[1] = tensor.shape[0];
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    const tensor_array = tensor.storage.array;
    var array = try allocator.alloc(T, tensor_array.len);
    errdefer allocator.free(array);
    var row: usize = 0;
    while (row < shape[0]) : (row += 1) {
        var column: usize = 0;
        while (column < shape[1]) : (column += 1) {
            const index = linear_index(stride, &[_]usize{ row, column });
            const tensor_index = linear_index(tensor.stride, &[_]usize{ column, row });
            array[index] = tensor_array[tensor_index];
        }
    }
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = array },
    };
}

test "transpose matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, [_][2]i32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const actual = try transpose(i32, &arena.allocator, tensor);
    const expected = try constant(&arena.allocator, [_][3]i32{
        .{ 1, 3, 5 },
        .{ 2, 4, 6 },
    });
    expectEqual(i32, actual, expected);
}

pub fn matrixMultiplyBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const x = context.forward_inputs[0];
    const y = context.forward_inputs[1];
    const outputs = try context.allocator.alloc(CpuTensor(T), 2);
    errdefer context.allocator.free(outputs);
    const x_transpose = try transpose(T, context.allocator, x);
    const y_transpose = try transpose(T, context.allocator, y);
    outputs[0] = try matrixMultiply(T, context.allocator, context.gradient_input, y_transpose);
    outputs[1] = try matrixMultiply(T, context.allocator, x_transpose, context.gradient_input);
    return outputs;
}

test "matrix multiply backward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][3]f64{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const y = try constant(&arena.allocator, [_][2]f64{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const gradient_input = try constant(&arena.allocator, [_][2]f64{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const actual = try matrixMultiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(&arena.allocator, [_][3]f64{
        .{ 3.75, 4.75, 5.75 },
        .{ 3.75, 4.75, 5.75 },
    });
    const expected_y_gradient = try constant(&arena.allocator, [_][2]f64{
        .{ 1.25, 1.25 },
        .{ 1.75, 1.75 },
        .{ 2.25, 2.25 },
    });
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}
