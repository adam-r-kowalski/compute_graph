const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const linearIndex = cpu_tensor.linearIndex;
const reduce = @import("reduce.zig").reduce;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const broadcast = @import("broadcast.zig");
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;

fn minimumScalar(comptime T: type) T {
    return switch (T) {
        f64 => std.math.f64_min,
        f32 => std.math.f32_min,
        f16 => std.math.f16_min,
        else => std.math.minInt(T),
    };
}

pub fn maximum(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), dimension: ?usize) !CpuTensor(T) {
    return try reduce(T, allocator, tensor, dimension, struct {
        fn call(accumulator: T, value: T) T {
            return std.math.max(accumulator, value);
        }
    }.call, minimumScalar(T));
}

test "maximum rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try maximum(f64, &arena.allocator, x, null);
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "maximum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 5, 10, 7, 8, 10 });
    const actual = try maximum(i32, &arena.allocator, x, null);
    const expected = try constant(i32, &arena.allocator, 10);
    expectEqual(i32, actual, expected);
}

test "maximum rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try maximum(f16, &arena.allocator, x, null);
    const expected = try constant(f16, &arena.allocator, 10);
    expectEqual(f16, actual, expected);
}

test "maximum rank 2 across 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try maximum(f16, &arena.allocator, x, 0);
    const expected = try constant(f16, &arena.allocator, .{ 5, 6 });
    expectEqual(f16, actual, expected);
}

test "maximum rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i8, &arena.allocator, .{
        .{
            .{ 5, 10 },
            .{ 7, 8 },
        },
        .{
            .{ 10, 8 },
            .{ 2, 6 },
        },
    });
    const actual = try maximum(i8, &arena.allocator, x, null);
    const expected = try constant(i8, &arena.allocator, 10);
    expectEqual(i8, actual, expected);
}

test "maximum rank 3 accross 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try maximum(i64, &arena.allocator, x, 0);
    const expected = try constant(i64, &arena.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    expectEqual(i64, actual, expected);
}

test "maximum rank 3 accross 1 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try maximum(i64, &arena.allocator, x, 1);
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, 4 },
        .{ 7, 8 },
    });
    expectEqual(i64, actual, expected);
}

test "maximum rank 3 accross 2 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try maximum(i64, &arena.allocator, x, 2);
    const expected = try constant(i64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    expectEqual(i64, actual, expected);
}

pub fn maximumBackward(comptime T: type, dimension: ?usize, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const allocator = context.allocator;

    const input = context.forward_inputs[0];
    const outputs = try allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    const shape = input.shape;
    const stride = input.stride;

    if (dimension) |d| {
        if (shape.len > 1) {
            const forward_input = input.storage.array;
            const array = try allocator.alloc(T, forward_input.len);
            errdefer allocator.free(array);

            const gradient_array = context.gradient_input.storage.array;

            var gradient_cartesian_index = try allocator.alloc(usize, shape.len - 1);
            defer allocator.free(gradient_cartesian_index);
            for (gradient_cartesian_index) |*e| e.* = 0;

            var array_cartesian_index = try allocator.alloc(usize, shape.len);
            defer allocator.free(array_cartesian_index);

            const gradient_shape = context.gradient_input.shape;

            const forward_output = context.forward_output.storage.array;

            while (true) {
                for (array_cartesian_index) |*e, i| {
                    if (i < d) {
                        e.* = gradient_cartesian_index[i];
                    } else if (i > d) {
                        e.* = gradient_cartesian_index[i - 1];
                    } else {
                        e.* = 0;
                    }
                }

                const gradient_linear_index = linearIndex(context.gradient_input.stride, gradient_cartesian_index);
                const gradient_value = gradient_array[gradient_linear_index];

                var i: usize = 0;
                while (i < shape[d]) {
                    array_cartesian_index[d] = i;
                    const array_linear_index = linearIndex(stride, array_cartesian_index);
                    const is_max = forward_input[array_linear_index] == forward_output[gradient_linear_index];
                    const contribution = if (is_max) gradient_value else 0;
                    array[array_linear_index] = contribution;
                    i += 1;
                }

                if (maximumCartesianIndex(gradient_shape, gradient_cartesian_index)) break;
                incrementCartesianIndex(gradient_shape, gradient_cartesian_index);
            }

            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = array },
            };
            return outputs;
        }
    }

    const gradient = context.gradient_input.storage.scalar;
    switch (input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = gradient },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            const forward_output = context.forward_output.storage.scalar;
            var count: T = 0;
            for (array) |e| {
                if (e == forward_output) count += 1;
            }
            const contribution = gradient / count;
            for (new_array) |*e, i| e.* = if (array[i] == forward_output) contribution else 0;
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "maximum backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, 4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, null);
    const actual = try maximumBackward(f64, null, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, null);
    const actual = try maximumBackward(f64, null, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 0, 0, 0, 1 });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 1 repeated max" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 5, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, null);
    const actual = try maximumBackward(f64, null, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 0.5, 0, 0, 0.5 });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 1 thrice repeated max" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 5, 5, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, null);
    const actual = try maximumBackward(f64, null, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 0.3333, 0.3333, 0, 0.3333 });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, null);
    const actual = try maximumBackward(f64, null, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 0 },
        .{ 0, 1 },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const forward_output = try maximum(f64, &arena.allocator, forward_input, 0);
    const actual = try maximumBackward(f64, 0, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0, 0.25 },
            .{ 0, 0 },
        },
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const forward_output = try maximum(f64, &arena.allocator, forward_input, 1);
    const actual = try maximumBackward(f64, 1, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.25 },
            .{ 0, 0 },
        },
        .{
            .{ 0, 0 },
            .{ 0.25, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const forward_output = try maximum(f64, &arena.allocator, forward_input, 2);
    const actual = try maximumBackward(f64, 2, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0, 0.25 },
            .{ 0, 0.25 },
        },
        .{
            .{ 0, 0.25 },
            .{ 0, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}
