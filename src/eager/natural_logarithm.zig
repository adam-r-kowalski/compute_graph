const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;
const mapBackward = @import("map.zig").mapBackward;

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
    return try mapBackward(T, context, struct {
        fn call(input: T, gradient: T) T {
            return 1 / input * gradient;
        }
    }.call);
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

test "naturalLogarithm rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f32, &leak_allocator.allocator, .{ 1, 2, 3, 4, 5, 6 });
    const actual = try naturalLogarithm(f32, &leak_allocator.allocator, x);
    defer actual.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    const expected = try constant(f32, &leak_allocator.allocator, .{
        0, 0.6931, 1.0986, 1.3862, 1.6094, 1.7917,
    });
    defer expected.deinit(&leak_allocator.allocator);
    expectEqual(f32, actual, expected);
}

test "gradient naturalLogarithm rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        0.2, 0.2, 0.2, 0.2, 0.2,
    });
    const forward_output = try naturalLogarithm(f64, &leak_allocator.allocator, x);
    const actual = try naturalLogarithmBackward(f64, backward.Context(f64){
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
        0.2, 0.1, 0.0666, 0.05, 0.04,
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}
