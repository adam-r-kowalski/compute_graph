const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn exponentiate(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = std.math.exp(scalar) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = std.math.exp(e);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "exponentiate rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try exponentiate(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, 0.0067));
    expectEqual(f64, actual, expected);
}

test "exponentiate rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f32{ 1, -2, 3, -4, -5, 6 });
    const actual = try exponentiate(f32, &arena.allocator, x);
    const expected = try constant(&arena.allocator, [_]f32{ 2.718, 0.135, 20.085, 0.001, 0.0006, 403.428 });
    expectEqual(f32, actual, expected);
}

test "exponentiate rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try exponentiate(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 2.718, 0.135 },
        .{ 20.085, 0.001 },
        .{ 0.0006, 403.428 },
    });
    expectEqual(f64, actual, expected);
}

pub fn exponentiateBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    switch (context.gradient_input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .scalar = std.math.exp(input.storage.scalar) * scalar },
            };
        },
        .array => |array| {
            const input_array = input.storage.array;
            var new_array = try context.allocator.alloc(T, input_array.len);
            for (new_array) |*e, i| e.* = std.math.exp(input_array[i]) * array[i];
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "exponentiate backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -4));
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try exponentiateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, @as(f64, 0.0183));
    expectEqual(f64, actual[0], expected);
}

test "exponentiate backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f64{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(&arena.allocator, [_]f64{ 2, 4, 6, 8, 10 });
    const actual = try exponentiateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, [_]f64{ 2.0, 29.556, 0.2987, 436.785, 0.0673 });
    expectEqual(f64, actual[0], expected);
}

test "exponentiate backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 0, -2 },
        .{ 3, -4 },
    });
    const gradient_input = try constant(&arena.allocator, [_][2]f64{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try exponentiateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 2.0, 0.5413 },
        .{ 120.513, 0.1465 },
    });
    expectEqual(f64, actual[0], expected);
}