const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;

pub fn naturalLogarithm(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    return try map(T, allocator, tensor, struct {
        fn call(t: T) T {
            return std.math.log(T, std.math.e, t);
        }
    }.call);
}

test "naturalLogarithm rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    const actual = try naturalLogarithm(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 1.6094);
    expectEqual(f64, actual, expected);
}

test "naturalLogarithm rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f32, &arena.allocator, .{ 1, 2, 3, 4, 5, 6 });
    const actual = try naturalLogarithm(f32, &arena.allocator, x);
    const expected = try constant(f32, &arena.allocator, .{ 0, 0.6931, 1.0986, 1.3862, 1.6094, 1.7917 });
    expectEqual(f32, actual, expected);
}

test "naturalLogarithm rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const actual = try naturalLogarithm(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 0.6931 },
        .{ 1.0986, 1.3862 },
        .{ 1.6094, 1.7917 },
    });
    expectEqual(f64, actual, expected);
}

pub fn naturalLogarithmBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    switch (context.gradient_input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .scalar = 1 / input.storage.scalar * scalar },
            };
        },
        .array => |array| {
            const input_array = input.storage.array;
            var new_array = try context.allocator.alloc(T, input_array.len);
            for (new_array) |*e, i| e.* = 1 / input_array[i] * array[i];
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "naturalLogarithm backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try naturalLogarithm(f64, &arena.allocator, x);
    const actual = try naturalLogarithmBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 0.25);
    expectEqual(f64, actual[0], expected);
}

test "naturalLogarithm backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 0.2, 0.2, 0.2, 0.2, 0.2 });
    const forward_output = try naturalLogarithm(f64, &arena.allocator, x);
    const actual = try naturalLogarithmBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0.2, 0.1, 0.0666, 0.05, 0.04 });
    expectEqual(f64, actual[0], expected);
}

test "naturalLogarithm backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 1. / 6., 1. / 6. },
        .{ 1. / 6., 1. / 6. },
        .{ 1. / 6., 1. / 6. },
    });
    const forward_output = try naturalLogarithm(f64, &arena.allocator, x);
    const actual = try naturalLogarithmBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0.1666, 0.0833 },
        .{ 0.0555, 0.0416 },
        .{ 0.0333, 0.0277 },
    });
    expectEqual(f64, actual[0], expected);
}
