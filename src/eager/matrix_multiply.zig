const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorLength = cpu_tensor.tensorLength;
const tensorStride = cpu_tensor.tensorStride;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

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
            const t_index = linearIndex(tensor.stride, &[_]usize{ i, j });
            tensor_array[t_index] = 0;
            var k: usize = 0;
            while (k < n) : (k += 1) {
                const x_index = linearIndex(x.stride, &[_]usize{ i, k });
                const y_index = linearIndex(y.stride, &[_]usize{ k, j });
                tensor_array[t_index] += x_array[x_index] * y_array[y_index];
            }
        }
    }
    return tensor;
}

test "matrixMultiply" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ 5, -6 },
    });
    const y = try constant(i32, &arena.allocator, .{
        .{ 1, -2, 3 },
        .{ -4, 5, -6 },
    });
    const actual = try matrixMultiply(i32, &arena.allocator, x, y);
    const expected = try constant(i32, &arena.allocator, .{
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
            const index = linearIndex(stride, &[_]usize{ row, column });
            const tensor_index = linearIndex(tensor.stride, &[_]usize{ column, row });
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
    const tensor = try constant(i32, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const actual = try transpose(i32, &arena.allocator, tensor);
    const expected = try constant(i32, &arena.allocator, .{
        .{ 1, 3, 5 },
        .{ 2, 4, 6 },
    });
    expectEqual(i32, actual, expected);
}

pub fn matrixMultiplyBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const allocator = context.allocator;
    const x = context.forward_inputs[0];
    const y = context.forward_inputs[1];
    const outputs = try allocator.alloc(CpuTensor(T), 2);
    errdefer allocator.free(outputs);
    const x_transpose = try transpose(T, allocator, x);
    defer x_transpose.deinit(allocator);
    const y_transpose = try transpose(T, allocator, y);
    defer y_transpose.deinit(allocator);
    outputs[0] = try matrixMultiply(T, allocator, context.gradient_input, y_transpose);
    outputs[1] = try matrixMultiply(T, allocator, x_transpose, context.gradient_input);
    return outputs;
}

test "matrix multiply backward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const y = try constant(f64, &arena.allocator, .{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const forward_output = try matrixMultiply(f64, &arena.allocator, x, y);
    const actual = try matrixMultiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, .{
        .{ 3.75, 4.75, 5.75 },
        .{ 3.75, 4.75, 5.75 },
    });
    const expected_y_gradient = try constant(f64, &arena.allocator, .{
        .{ 1.25, 1.25 },
        .{ 1.75, 1.75 },
        .{ 2.25, 2.25 },
    });
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "matrix multiply rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(i32, &leak_allocator.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ 5, -6 },
    });
    const y = try constant(i32, &leak_allocator.allocator, .{
        .{ 1, -2, 3 },
        .{ -4, 5, -6 },
    });
    const actual = try matrixMultiply(i32, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    const expected = try constant(i32, &leak_allocator.allocator, .{
        .{ 9, -12, 15 },
        .{ 19, -26, 33 },
        .{ 29, -40, 51 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    expectEqual(i32, actual, expected);
}

test "matrix multiply backward" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const forward_output = try matrixMultiply(f64, &leak_allocator.allocator, x, y);
    const actual = try matrixMultiplyBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected_x_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ 3.75, 4.75, 5.75 },
        .{ 3.75, 4.75, 5.75 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ 1.25, 1.25 },
        .{ 1.75, 1.75 },
        .{ 2.25, 2.25 },
    });
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}
