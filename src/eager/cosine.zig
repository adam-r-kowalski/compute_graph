const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;
const mapBackward = @import("map.zig").mapBackward;

pub fn cosine(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    return try map(T, allocator, tensor, struct {
        fn call(t: T) T {
            return std.math.cos(t);
        }
    }.call);
}

test "cosine rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try cosine(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 0.2836);
    expectEqual(f64, actual, expected);
}

test "cosine rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try cosine(f32, &arena.allocator, x);
    const expected = try constant(f32, &arena.allocator, .{ 0.5403, -0.4161, -0.9899, -0.6536, 0.2836, 0.9601 });
    expectEqual(f32, actual, expected);
}

test "cosine rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try cosine(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0.5403, -0.4161 },
        .{ -0.9899, -0.6536 },
        .{ 0.2836, 0.96017 },
    });
    expectEqual(f64, actual, expected);
}

pub fn cosineBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    return try mapBackward(T, context, struct {
        fn call(input: T, gradient: T) T {
            return -std.math.sin(input) * gradient;
        }
    }.call);
}

test "cosine backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try cosine(f64, &arena.allocator, x);
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, -0.75680);
    expectEqual(f64, actual[0], expected);
}

test "cosine backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 0, 2, -3, 4, -5 });
    const forward_output = try cosine(f64, &arena.allocator, x);
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, -3.6371, 0.8467, 6.0544, -9.5892 });
    expectEqual(f64, actual[0], expected);
}

test "cosine backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 0, -2 },
        .{ 3, -4 },
    });
    const forward_output = try cosine(f64, &arena.allocator, x);
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 3.6371 },
        .{ -0.8467, -6.0544 },
    });
    expectEqual(f64, actual[0], expected);
}

test "cosine rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f32, &leak_allocator.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try cosine(f32, &leak_allocator.allocator, x);
    defer actual.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    const expected = try constant(f32, &leak_allocator.allocator, .{
        0.5403, -0.4161, -0.9899, -0.6536, 0.2836, 0.9601,
    });
    defer expected.deinit(&leak_allocator.allocator);
    expectEqual(f32, actual, expected);
}

test "gradient cosine rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{ 2, 4, 6, 8, 10 });
    const forward_output = try cosine(f64, &leak_allocator.allocator, x);
    const actual = try cosineBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected = try constant(f64, &leak_allocator.allocator, .{
        0, -3.6371, 0.8467, 6.0544, -9.5892,
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}
