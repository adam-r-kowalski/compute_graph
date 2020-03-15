const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const broadcast = @import("broadcast.zig");
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;
const zeroBroadcastedIndex = broadcast.zeroBroadcastedIndex;
const reduce = @import("reduce.zig").reduce;
const ReduceParameters = @import("reduce.zig").ReduceParameters;

pub fn sum(
    comptime T: type,
    allocator: *Allocator,
    tensor: CpuTensor(T),
    parameters: ReduceParameters,
) !CpuTensor(T) {
    return try reduce(T, allocator, tensor, struct {
        fn call(accumulator: T, value: T) T {
            return accumulator + value;
        }
    }.call, 0, parameters);
}

test "sum rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try sum(f64, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "sum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 5, 10, 7, 8, 10 });
    const actual = try sum(i32, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(i32, &arena.allocator, 40);
    expectEqual(i32, actual, expected);
}

test "sum rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try sum(f16, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(f16, &arena.allocator, 48);
    expectEqual(f16, actual, expected);
}

test "sum rank 2 across 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try sum(f16, &arena.allocator, x, ReduceParameters{ .dimension = 0 });
    const expected = try constant(f16, &arena.allocator, .{ 3, 12 });
    expectEqual(f16, actual, expected);
}

test "sum rank 3" {
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
    const actual = try sum(i8, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(i8, &arena.allocator, 56);
    expectEqual(i8, actual, expected);
}

test "sum rank 3 accross 0 dimension" {
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
    const actual = try sum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 0 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 6, 8 },
        .{ 4, 12 },
    });
    expectEqual(i64, actual, expected);
}

test "sum rank 3 accross 1 dimension" {
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
    const actual = try sum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 1 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ -2, 6 },
        .{ 12, 14 },
    });
    expectEqual(i64, actual, expected);
}

test "sum rank 3 accross 2 dimension" {
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
    const actual = try sum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 2 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 3, 1 },
        .{ 11, 15 },
    });
    expectEqual(i64, actual, expected);
}

test "sum keep dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try sum(i64, &arena.allocator, x, ReduceParameters{ .keep_dimensions = true });
    const expected = try constant(i64, &arena.allocator, .{
        .{21},
    });
    expectEqual(i64, actual, expected);
}

test "sum keep dimensions 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try sum(i64, &arena.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 0,
    });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 5, 7, 9 },
    });
    expectEqual(i64, actual, expected);
}

test "sum keep dimensions 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try sum(i64, &arena.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 1,
    });
    const expected = try constant(i64, &arena.allocator, .{
        .{6}, .{15},
    });
    expectEqual(i64, actual, expected);
}

pub fn sumBackward(comptime T: type, parameters: ReduceParameters, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const allocator = context.allocator;

    const input = context.forward_inputs[0];
    const outputs = try allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    const shape = input.shape;
    const stride = input.stride;

    if (parameters.dimension) |d| {
        if (shape.len > 1) {
            const array = try allocator.alloc(T, input.storage.array.len);
            errdefer allocator.free(array);

            const gradient_array = context.gradient_input.storage.array;

            var gradient_cartesian_index = try allocator.alloc(usize, shape.len - 1);
            defer allocator.free(gradient_cartesian_index);
            for (gradient_cartesian_index) |*e| e.* = 0;

            var array_cartesian_index = try allocator.alloc(usize, shape.len);
            defer allocator.free(array_cartesian_index);

            const gradient_shape = context.gradient_input.shape;

            while (true) {
                zeroBroadcastedIndex(gradient_cartesian_index, d, array_cartesian_index);
                const gradient_linear_index = linearIndex(context.gradient_input.stride, gradient_cartesian_index);

                var i: usize = 0;
                while (i < shape[d]) {
                    array_cartesian_index[d] = i;
                    const array_linear_index = linearIndex(stride, array_cartesian_index);
                    array[array_linear_index] = gradient_array[gradient_linear_index];
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
            for (new_array) |*e, i| e.* = gradient;
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "sum backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, 4);
    const parameters = ReduceParameters{};
    const forward_output = try sum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try sumBackward(f64, parameters, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const parameters = ReduceParameters{};
    const forward_output = try sum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try sumBackward(f64, parameters, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 1, 1, 1, 1, 1 });
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const parameters = ReduceParameters{};
    const forward_output = try sum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try sumBackward(f64, parameters, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 1, 1 },
        .{ 1, 1 },
    });
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 3 dimension 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const parameters = ReduceParameters{ .dimension = 0 };
    const forward_output = try sum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.5 },
        .{ 0.75, 1 },
    });
    const actual = try sumBackward(f64, parameters, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.5 },
            .{ 0.75, 1 },
        },
        .{
            .{ 0.25, 0.5 },
            .{ 0.75, 1 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 3 dimension 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const parameters = ReduceParameters{ .dimension = 1 };
    const forward_output = try sum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.5 },
        .{ 0.75, 1 },
    });
    const actual = try sumBackward(f64, parameters, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.5 },
            .{ 0.25, 0.5 },
        },
        .{
            .{ 0.75, 1 },
            .{ 0.75, 1 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 3 dimension 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const parameters = ReduceParameters{ .dimension = 2 };
    const forward_output = try sum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.5 },
        .{ 0.75, 1 },
    });
    const actual = try sumBackward(f64, parameters, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.25 },
            .{ 0.5, 0.5 },
        },
        .{
            .{ 0.75, 0.75 },
            .{ 1, 1 },
        },
    });
    expectEqual(f64, actual[0], expected);
}
