const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn cosine(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = std.math.cos(scalar) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = std.math.cos(e);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "cosine rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try cosine(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, 0.2836));
    expectEqual(f64, actual, expected);
}

test "cosine rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f32{ 1, -2, 3, -4, -5, 6 });
    const actual = try cosine(f32, &arena.allocator, x);
    const expected = try constant(&arena.allocator, [_]f32{ 0.5403, -0.4161, -0.9899, -0.6536, 0.2836, 0.9601 });
    expectEqual(f32, actual, expected);
}

test "cosine rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try cosine(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 0.5403, -0.4161 },
        .{ -0.9899, -0.6536 },
        .{ 0.2836, 0.96017 },
    });
    expectEqual(f64, actual, expected);
}

pub fn cosineBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    switch (context.gradient_input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .scalar = -std.math.sin(input.storage.scalar) * scalar },
            };
        },
        .array => |array| {
            const input_array = input.storage.array;
            var new_array = try context.allocator.alloc(T, input_array.len);
            for (new_array) |*e, i| e.* = -std.math.sin(input_array[i]) * array[i];
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "cosine backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -4));
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, @as(f64, -0.75680));
    expectEqual(f64, actual[0], expected);
}

test "cosine backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f64{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(&arena.allocator, [_]f64{ 2, 4, 6, 8, 10 });
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, [_]f64{ 0, -3.6371, 0.8467, 6.0544, -9.5892 });
    expectEqual(f64, actual[0], expected);
}

test "cosine backward rank 2" {
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
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 0, 3.6371 },
        .{ -0.8467, -6.0544 },
    });
    expectEqual(f64, actual[0], expected);
}