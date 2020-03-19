const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;
const mapBackward = @import("map.zig").mapBackward;

pub fn exponentiate(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    return try map(T, allocator, tensor, struct {
        fn call(input: T) T {
            return std.math.exp(input);
        }
    }.call);
}

test "exponentiate rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try exponentiate(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 0.0067);
    expectEqual(f64, actual, expected);
}

test "exponentiate rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try exponentiate(f32, &arena.allocator, x);
    const expected = try constant(f32, &arena.allocator, .{ 2.7182, 0.1353, 20.0855, 0.0183, 0.00673, 403.4288 });
    expectEqual(f32, actual, expected);
}

test "exponentiate rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try exponentiate(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, .{
        .{ 2.7182, 0.1353 },
        .{ 20.0855, 0.0183 },
        .{ 0.0067, 403.4287 },
    });
    expectEqual(f64, actual, expected);
}

pub fn exponentiateBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    return try mapBackward(T, context, struct {
        fn call(input: T, gradient: T) T {
            return std.math.exp(input) * gradient;
        }
    }.call);
}

test "exponentiate backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try exponentiate(f64, &arena.allocator, x);
    const actual = try exponentiateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 0.0183);
    expectEqual(f64, actual[0], expected);
}

test "exponentiate backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const forward_output = try exponentiate(f64, &arena.allocator, x);
    const actual = try exponentiateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 2.0, 29.5562, 0.2987, 436.7852, 0.0673 });
    expectEqual(f64, actual[0], expected);
}

test "exponentiate backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 0, -2 },
        .{ 3, -4 },
    });
    const forward_output = try exponentiate(f64, &arena.allocator, x);
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try exponentiateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 2.0, 0.5413 },
        .{ 120.5132, 0.1465 },
    });
    expectEqual(f64, actual[0], expected);
}
