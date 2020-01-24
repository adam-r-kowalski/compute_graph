const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

fn TensorType(comptime T: type) type {
    const ScalarType = switch (T.ScalarType) {
        f64, f32, f16 => T.ScalarType,
        i64 => f64,
        i32 => f32,
        i8 => f16,
        else => @compileError("ScalarType not supported"),
    };
    return CpuTensor(ScalarType);
}

fn coerceToFloat(comptime T: type, x: var) T {
    return switch (@TypeOf(x)) {
        f64, f32, f16 => @as(T, x),
        else => @intToFloat(T, x),
    };
}

pub fn mean(allocator: *Allocator, tensor: var) !TensorType(@TypeOf(tensor)) {
    const T = TensorType(@TypeOf(tensor));
    const shape = try allocator.alloc(usize, 0);
    errdefer allocator.free(shape);
    const stride = try allocator.alloc(usize, 0);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return T{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = coerceToFloat(T.ScalarType, scalar) },
            };
        },
        .array => |array| {
            var sum: T.ScalarType = 0;
            for (array) |e| sum += coerceToFloat(T.ScalarType, e);
            return T{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = sum / coerceToFloat(T.ScalarType, array.len) },
            };
        },
    }
}

test "mean rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, -5));
    expectEqual(actual, expected);
}

test "mean rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{ 5, 10, 7, 8, 10 });
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f32, 8));
    expectEqual(actual, expected);
}

test "mean rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f16, 8));
    expectEqual(actual, expected);
}

test "mean rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 5, 10 },
            .{ 7, 8 },
        },
        .{
            .{ 10, 8 },
            .{ 2, 6 },
        },
    });
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f16, 7));
    expectEqual(actual, expected);
}

pub fn mean_backward(context: var) !@TypeOf(context.forward_inputs) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(@TypeOf(input), 1);
    errdefer context.allocator.free(outputs);
    const scalar = context.gradient_input.storage.scalar;
    // TODO(Adam) length should be factored into seperate function
    //            returning the number of scalars in the tensor
    //            const len = length(context.forward_inputs[0]);
    const len = input.storage.array.len;
    const value = scalar / @intToFloat(@TypeOf(scalar), len);
    // TODO(Adam) build a function which creates a tensor given a value and a shape
    //            const outputs[0] = fill(value, context.forward_inputs[0].shape);
    outputs[0] = input;
    return outputs;
}

test "mean backward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try mean(&arena.allocator, x);
    const actual = try mean_backward(backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = y,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    expectEqual(actual[0], x);
}
