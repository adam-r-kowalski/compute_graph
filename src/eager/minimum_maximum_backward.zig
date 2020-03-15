const std = @import("std");
const Context = @import("backward.zig").Context;
const ReduceParameters = @import("reduce.zig").ReduceParameters;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const linearIndex = cpu_tensor.linearIndex;
const broadcast = @import("broadcast.zig");
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;
const zeroBroadcastedIndex = broadcast.zeroBroadcastedIndex;
const zeroBroadcastedIndexKeepDimension = broadcast.zeroBroadcastedIndexKeepDimension;

fn backwardAcrossDimension(comptime T: type, dimension: usize, context: Context(T)) ![]CpuTensor(T) {
    const allocator = context.allocator;
    const input = context.forward_inputs[0];
    const shape = input.shape;
    const stride = input.stride;
    const forward_input = input.storage.array;
    const outputs = try allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);
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
        const gradient_linear_index = linearIndex(context.gradient_input.stride, gradient_cartesian_index);
        const value = forward_output[gradient_linear_index];
        zeroBroadcastedIndex(gradient_cartesian_index, dimension, array_cartesian_index);
        var i: usize = 0;
        var count: T = 0;
        while (i < shape[dimension]) {
            array_cartesian_index[dimension] = i;
            const array_linear_index = linearIndex(stride, array_cartesian_index);
            if (forward_input[array_linear_index] == value) count += 1;
            i += 1;
        }

        zeroBroadcastedIndex(gradient_cartesian_index, dimension, array_cartesian_index);
        const contribution = gradient_array[gradient_linear_index] / count;
        i = 0;
        while (i < shape[dimension]) {
            array_cartesian_index[dimension] = i;
            const array_linear_index = linearIndex(stride, array_cartesian_index);
            const forward_value = forward_input[array_linear_index];
            array[array_linear_index] = if (forward_value == value) contribution else 0;
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

fn backwardAcrossKeepDimensions(comptime T: type, dimension: usize, context: Context(T)) ![]CpuTensor(T) {
    const allocator = context.allocator;
    const input = context.forward_inputs[0];
    const shape = input.shape;
    const stride = input.stride;
    const forward_input = input.storage.array;
    const outputs = try allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);
    const array = try allocator.alloc(T, forward_input.len);
    errdefer allocator.free(array);

    const gradient_array = context.gradient_input.storage.array;

    var gradient_cartesian_index = try allocator.alloc(usize, shape.len);
    defer allocator.free(gradient_cartesian_index);
    for (gradient_cartesian_index) |*e| e.* = 0;

    var array_cartesian_index = try allocator.alloc(usize, shape.len);
    defer allocator.free(array_cartesian_index);

    const gradient_shape = context.gradient_input.shape;

    const forward_output = context.forward_output.storage.array;

    while (true) {
        const gradient_linear_index = linearIndex(context.gradient_input.stride, gradient_cartesian_index);
        const value = forward_output[gradient_linear_index];
        zeroBroadcastedIndexKeepDimension(gradient_cartesian_index, dimension, array_cartesian_index);
        var i: usize = 0;
        var count: T = 0;
        while (i < shape[dimension]) {
            array_cartesian_index[dimension] = i;
            const array_linear_index = linearIndex(stride, array_cartesian_index);
            if (forward_input[array_linear_index] == value) count += 1;
            i += 1;
        }

        zeroBroadcastedIndexKeepDimension(gradient_cartesian_index, dimension, array_cartesian_index);
        const contribution = gradient_array[gradient_linear_index] / count;
        i = 0;
        while (i < shape[dimension]) {
            array_cartesian_index[dimension] = i;
            const array_linear_index = linearIndex(stride, array_cartesian_index);
            const forward_value = forward_input[array_linear_index];
            array[array_linear_index] = if (forward_value == value) contribution else 0;
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

pub fn backward(comptime T: type, parameters: ReduceParameters, context: Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const shape = input.shape;

    if (parameters.dimension) |d| {
        if (shape.len > 1) {
            if (parameters.keep_dimensions)
                return try backwardAcrossKeepDimensions(T, d, context);
            return try backwardAcrossDimension(T, d, context);
        }
    }

    const allocator = context.allocator;
    const outputs = try allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);
    const stride = input.stride;
    switch (input.storage) {
        .scalar => |scalar| {
            const gradient = context.gradient_input.storage.scalar;
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = gradient },
            };
        },
        .array => |array| {
            const gradient = switch (context.gradient_input.storage) {
                .scalar => |s| s,
                .array => |a| a[0],
            };
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            const forward_output = switch (context.forward_output.storage) {
                .scalar => |s| s,
                .array => |a| a[0],
            };
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
