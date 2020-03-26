const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const copy = cpu_tensor.copy;
const invoke = @import("invoke.zig").invoke;
const backward = @import("backward.zig");

pub fn map(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), invokable: var) !CpuTensor(T) {
    const shape = try copy(usize, allocator, tensor.shape);
    errdefer allocator.free(shape);
    const stride = try copy(usize, allocator, tensor.stride);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = invoke(invokable, .{scalar}) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = invoke(invokable, .{e});
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

pub fn mapBackward(comptime T: type, context: backward.Context(T), invokable: var) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const allocator = context.allocator;
    const input = context.forward_inputs[0];
    const outputs = try allocator.alloc(CpuTensor(T), 1);
    errdefer allocator.free(outputs);
    switch (context.gradient_input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .scalar = invoke(invokable, .{ input.storage.scalar, scalar }) },
            };
        },
        .array => |array| {
            const input_array = input.storage.array;
            var new_array = try allocator.alloc(T, input_array.len);
            errdefer allocator.free(new_array);
            for (new_array) |*e, i| e.* = invoke(invokable, .{ input_array[i], array[i] });
            const shape = try copy(usize, allocator, input.shape);
            errdefer allocator.free(shape);
            const stride = try copy(usize, allocator, input.stride);
            errdefer allocator.free(stride);
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}
