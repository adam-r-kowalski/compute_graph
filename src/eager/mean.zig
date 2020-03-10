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
    const ScalarType = switch (T) {
        f64, f32, f16 => T,
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

// TODO(enhancement) mean should probably work across a particular dimension
// TODO(refactor) mean should not be possible for int scalar type, implicit casting is a bad idea
pub fn mean(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !TensorType(T) {
    const Tensor = TensorType(T);
    const shape = try allocator.alloc(usize, 0);
    errdefer allocator.free(shape);
    const stride = try allocator.alloc(usize, 0);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return Tensor{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = coerceToFloat(Tensor.ScalarType, scalar) },
            };
        },
        .array => |array| {
            var sum: Tensor.ScalarType = 0;
            for (array) |e| sum += coerceToFloat(Tensor.ScalarType, e);
            return Tensor{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = sum / coerceToFloat(Tensor.ScalarType, array.len) },
            };
        },
    }
}

test "mean rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try mean(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "mean rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 5, 10, 7, 8, 10 });
    const actual = try mean(i32, &arena.allocator, x);
    const expected = try constant(f32, &arena.allocator, 8);
    expectEqual(f32, actual, expected);
}

test "mean rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try mean(f16, &arena.allocator, x);
    const expected = try constant(f16, &arena.allocator, 8);
    expectEqual(f16, actual, expected);
}

test "mean rank 3" {
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
    const actual = try mean(i8, &arena.allocator, x);
    const expected = try constant(f16, &arena.allocator, 7);
    expectEqual(f16, actual, expected);
}

fn length(comptime T: type, tensor: CpuTensor(T)) usize {
    return switch (tensor.storage) {
        .scalar => 1,
        .array => |array| array.len,
    };
}

test "length rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    std.testing.expectEqual(length(f64, x), 1);
}

test "length rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3 });
    std.testing.expectEqual(length(f64, x), 3);
}

test "length rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    std.testing.expectEqual(length(f64, x), 4);
}

fn fill(comptime T: type, allocator: *Allocator, literal: T, shape: []const usize) !CpuTensor(T) {
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
    const actual = try fill(f64, &arena.allocator, 0.15, &[_]usize{});
    const expected = try constant(f64, &arena.allocator, 0.15);
    expectEqual(f64, actual, expected);
}

test "fill vector" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try fill(f64, &arena.allocator, 0.5, &[_]usize{3});
    const expected = try constant(f64, &arena.allocator, .{ 0.5, 0.5, 0.5 });
    expectEqual(f64, actual, expected);
}

test "fill matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try fill(f64, &arena.allocator, 5, &[_]usize{ 2, 3 });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 5, 5, 5 },
        .{ 5, 5, 5 },
    });
    expectEqual(f64, actual, expected);
}

pub fn meanBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);
    const scalar = context.gradient_input.storage.scalar;
    const value = scalar / @intToFloat(T, length(T, input));
    outputs[0] = try fill(T, context.allocator, value, context.forward_inputs[0].shape);
    return outputs;
}

test "mean backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, 4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try mean(f64, &arena.allocator, forward_input);
    const actual = try meanBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
}

test "mean backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, @as(f64, 1));
    const forward_output = try mean(f64, &arena.allocator, forward_input);
    const actual = try meanBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0.2, 0.2, 0.2, 0.2, 0.2 });
    expectEqual(f64, actual[0], expected);
}

test "mean backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try mean(f64, &arena.allocator, forward_input);
    const actual = try meanBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    expectEqual(f64, actual[0], expected);
}
