const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const copy = cpu_tensor.copy;
const CpuTensor = cpu_tensor.CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const map = @import("map.zig").map;
const mapBackward = @import("map.zig").mapBackward;

fn absoluteScalar(comptime T: type, x: T) error{Overflow}!T {
    return switch (T) {
        f64, f32, f16 => std.math.absFloat(x),
        i64, i32, i8 => try std.math.absInt(x),
        else => @compileError("ScalarType not supported"),
    };
}

pub fn absolute(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    // TODO: make this work
    // return try map(T, allocator, tensor, struct {
    //     fn call(input: T) T {
    //         return try absoluteScalar(T, input);
    //     }
    // }.call);
    const shape = try copy(usize, allocator, tensor.shape);
    errdefer allocator.free(shape);
    const stride = try copy(usize, allocator, tensor.stride);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = try absoluteScalar(T, scalar) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = try absoluteScalar(T, e);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "absolute rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try absolute(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 5);
    expectEqual(f64, actual, expected);
}

test "absolute rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try absolute(i32, &arena.allocator, x);
    const expected = try constant(i32, &arena.allocator, .{ 1, 2, 3, 4, 5, 6 });
    expectEqual(i32, actual, expected);
}

test "absolute rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try absolute(f16, &arena.allocator, x);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f16, actual, expected);
}

test "absolute rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const actual = try absolute(i8, &arena.allocator, x);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ 3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    expectEqual(i8, actual, expected);
}

pub fn absoluteBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    return mapBackward(T, context, struct {
        fn call(input: T, gradient: T) T {
            if (input > 0)
                return gradient;
            if (input < 0)
                return -gradient;
            return 0;
        }
    }.call);
}

test "absolute backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try absolute(f64, &arena.allocator, x);
    const actual = try absoluteBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, -1);
    expectEqual(f64, actual[0], expected);
}

test "absolute backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const forward_output = try absolute(f64, &arena.allocator, x);
    const actual = try absoluteBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 4, -6, 8, -10 });
    expectEqual(f64, actual[0], expected);
}

test "absolute backward rank 2" {
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
    const forward_output = try absolute(f64, &arena.allocator, x);
    const actual = try absoluteBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, -4 },
        .{ 6, -8 },
    });
    expectEqual(f64, actual[0], expected);
}

test "absolute rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(i32, &leak_allocator.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try absolute(i32, &leak_allocator.allocator, x);
    defer actual.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    const expected = try constant(i32, &leak_allocator.allocator, .{ 1, 2, 3, 4, 5, 6 });
    defer expected.deinit(&leak_allocator.allocator);
    expectEqual(i32, actual, expected);
}

test "gradient absolute rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{ 2, 4, 6, 8, 10 });
    const forward_output = try absolute(f64, &leak_allocator.allocator, x);
    const actual = try absoluteBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected = try constant(f64, &leak_allocator.allocator, .{ 0, 4, -6, 8, -10 });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}
