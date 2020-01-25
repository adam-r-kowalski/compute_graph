const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const CpuStorage = cpu_tensor.CpuStorage;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
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

fn length(tensor: var) usize {
    return switch (tensor.storage) {
        .scalar => 1,
        .array => |array| array.len,
    };
}

test "length rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, 5));
    std.testing.expectEqual(length(x), 1);
}

test "length rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f64{ 1, 2, 3 });
    std.testing.expectEqual(length(x), 3);
}

test "length rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    std.testing.expectEqual(length(x), 4);
}

fn fill(allocator: *Allocator, literal: var, shape: []const usize) !CpuTensor(@TypeOf(literal)) {
    const T = @TypeOf(literal);
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    if (shape.len == 0) {
        return CpuTensor(T){
            .shape = shape,
            .stride = stride,
            .storage = CpuStorage(T){ .scalar = literal },
        };
    }
    var array = try allocator.alloc(T, tensorLength(shape));
    errdefer allocator.free(array);
    for (array) |*e| e.* = literal;
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = CpuStorage(T){ .array = array },
    };
}

test "fill scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try fill(&arena.allocator, @as(f64, 0.15), &[_]usize{});
    const expected = try constant(&arena.allocator, @as(f64, 0.15));
    expectEqual(actual, expected);
}

test "fill vector" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try fill(&arena.allocator, @as(f64, 0.5), &[_]usize{3});
    const expected = try constant(&arena.allocator, [_]f64{ 0.5, 0.5, 0.5 });
    expectEqual(actual, expected);
}

test "fill matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try fill(&arena.allocator, @as(f64, 5), &[_]usize{ 2, 3 });
    const expected = try constant(&arena.allocator, [_][3]f64{
        .{ 5, 5, 5 },
        .{ 5, 5, 5 },
    });
    expectEqual(actual, expected);
}

pub fn mean_backward(context: var) ![]@TypeOf(context.gradient_input) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(@TypeOf(input), 1);
    errdefer context.allocator.free(outputs);
    const scalar = context.gradient_input.storage.scalar;
    const value = scalar / @intToFloat(@TypeOf(scalar), length(input));
    outputs[0] = try fill(context.allocator, value, context.forward_inputs[0].shape);
    return outputs;
}

test "mean backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(&arena.allocator, @as(f64, 4));
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try mean_backward(backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
    });
    const expected = try constant(&arena.allocator, @as(f64, 1));
    expectEqual(actual[0], expected);
}

test "mean backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(&arena.allocator, [_]f64{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try mean_backward(backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
    });
    const expected = try constant(&arena.allocator, [_]f64{ 0.2, 0.2, 0.2, 0.2, 0.2 });
    expectEqual(actual[0], expected);
}

test "mean backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try mean_backward(backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
    });
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    expectEqual(actual[0], expected);
}
