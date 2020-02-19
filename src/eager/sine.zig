const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn sine(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = std.math.sin(scalar) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = std.math.sin(e);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "sine rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try sine(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, 0.95892));
    expectEqual(f64, actual, expected);
}

test "sine rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f32{ 1, -2, 3, -4, -5, 6 });
    const actual = try sine(f32, &arena.allocator, x);
    const expected = try constant(&arena.allocator, [_]f32{ 0.84147, -0.90929, 0.14112, 0.7568, 0.9589, -0.27941 });
    expectEqual(f32, actual, expected);
}

test "sine rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try sine(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 0.84147, -0.90929 },
        .{ 0.14112, 0.7568 },
        .{ 0.9589, -0.27941 },
    });
    expectEqual(f64, actual, expected);
}

pub fn sineBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    switch (context.gradient_input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .scalar = std.math.cos(input.storage.scalar) * scalar },
            };
        },
        .array => |array| {
            const input_array = input.storage.array;
            var new_array = try context.allocator.alloc(T, input_array.len);
            for (new_array) |*e, i| e.* = std.math.cos(input_array[i]) * array[i];
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "sine backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -4));
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try sineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, @as(f64, -0.65364));
    expectEqual(f64, actual[0], expected);
}

test "sine backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f64{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(&arena.allocator, [_]f64{ 2, 4, 6, 8, 10 });
    const actual = try sineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, [_]f64{ 2, -1.6645, -5.9399, -5.2291, 2.8366 });
    expectEqual(f64, actual[0], expected);
}

test "sine backward rank 2" {
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
    const actual = try sineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 2, -1.6645 },
        .{ -5.9399, -5.2291 },
    });
    expectEqual(f64, actual[0], expected);
}
