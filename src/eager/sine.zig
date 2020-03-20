const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;
const mapBackward = @import("map.zig").mapBackward;

pub fn sine(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    return try map(T, allocator, tensor, struct {
        fn call(t: T) T {
            return std.math.sin(t);
        }
    }.call);
}

test "sine rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try sine(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 0.95892);
    expectEqual(f64, actual, expected);
}

test "sine rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try sine(f32, &arena.allocator, x);
    const expected = try constant(f32, &arena.allocator, .{ 0.84147, -0.90929, 0.14112, 0.7568, 0.9589, -0.27941 });
    expectEqual(f32, actual, expected);
}

test "sine rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try sine(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0.84147, -0.90929 },
        .{ 0.14112, 0.7568 },
        .{ 0.9589, -0.27941 },
    });
    expectEqual(f64, actual, expected);
}

pub fn sineBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    return try mapBackward(T, context, struct {
        fn call(input: T, gradient: T) T {
            return std.math.cos(input) * gradient;
        }
    }.call);
}

test "sine backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try sine(f64, &arena.allocator, x);
    const actual = try sineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, -0.65364);
    expectEqual(f64, actual[0], expected);
}

test "sine backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const forward_output = try sine(f64, &arena.allocator, x);
    const actual = try sineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 2, -1.6645, -5.9399, -5.2291, 2.8366 });
    expectEqual(f64, actual[0], expected);
}

test "sine backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 0, -2 },
        .{ 3, -4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const forward_output = try sine(f64, &arena.allocator, x);
    const actual = try sineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 2, -1.6645 },
        .{ -5.9399, -5.2291 },
    });
    expectEqual(f64, actual[0], expected);
}
