const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;
const mapBackward = @import("map.zig").mapBackward;

pub fn power(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), n: var) !CpuTensor(T) {
    const Closure = struct {
        n_: T,

        pub fn call(self: @This(), input: T) T {
            return std.math.pow(T, input, self.n_);
        }
    };
    return try map(T, allocator, tensor, Closure{ .n_ = n });
}

test "power rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, -5);
    const actual = try power(i64, &arena.allocator, x, 2);
    const expected = try constant(i64, &arena.allocator, 25);
    expectEqual(i64, actual, expected);
}

test "power rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try power(f32, &arena.allocator, x, 3);
    const expected = try constant(f32, &arena.allocator, .{ 1, -8, 27, -64, -125, 216 });
    expectEqual(f32, actual, expected);
}

test "power rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try power(f64, &arena.allocator, x, -2);
    const expected = try constant(f64, &arena.allocator, .{
        .{ 1, 0.25 },
        .{ 0.1111, 0.0625 },
        .{ 0.04, 0.0277 },
    });
    expectEqual(f64, actual, expected);
}

test "power rank 2 float" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const actual = try power(f64, &arena.allocator, x, 2.5);
    const expected = try constant(f64, &arena.allocator, .{
        .{ 1, 5.6568 },
        .{ 15.5884, 32 },
        .{ 55.9016, 88.1816 },
    });
    expectEqual(f64, actual, expected);
}

pub fn powerBackward(comptime T: type, n: var, context: backward.Context(T)) ![]CpuTensor(T) {
    const Closure = struct {
        n_: T,

        pub fn call(self: @This(), input: T, gradient: T) T {
            return self.n_ * std.math.pow(T, input, self.n_ - 1.0) * gradient;
        }
    };
    return try mapBackward(T, context, Closure{ .n_ = n });
}

test "power backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const n = 2;
    const forward_output = try power(f64, &arena.allocator, x, n);
    const actual = try powerBackward(f64, n, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, -8);
    expectEqual(f64, actual[0], expected);
}

test "power backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 0.2, 0.2, 0.2, 0.2, 0.2 });
    const n = 3;
    const forward_output = try power(f64, &arena.allocator, x, n);
    const actual = try powerBackward(f64, n, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 2.4, 5.4, 9.6, 15 });
    expectEqual(f64, actual[0], expected);
}

test "power backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const n = 2.5;
    const forward_output = try power(f64, &arena.allocator, x, n);
    const actual = try powerBackward(f64, n, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0.625, 1.7677 },
        .{ 3.2475, 5 },
    });
    expectEqual(f64, actual[0], expected);
}

test "power rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f32, &leak_allocator.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try power(f32, &leak_allocator.allocator, x, 3);
    defer actual.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    const expected = try constant(f32, &leak_allocator.allocator, .{ 1, -8, 27, -64, -125, 216 });
    defer expected.deinit(&leak_allocator.allocator);
    expectEqual(f32, actual, expected);
}

test "gradient power rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{ 0, 2, -3, 4, -5 });
    const n = 3;
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        0.2, 0.2, 0.2, 0.2, 0.2,
    });
    const forward_output = try power(f64, &leak_allocator.allocator, x, n);
    const actual = try powerBackward(f64, n, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected = try constant(f64, &leak_allocator.allocator, .{ 0, 2.4, 5.4, 9.6, 15 });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}
