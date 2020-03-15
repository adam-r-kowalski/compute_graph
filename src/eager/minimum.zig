const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const ReduceParameters = @import("reduce.zig").ReduceParameters;
const reduce = @import("reduce.zig").reduce;
const expectEqual = @import("../testing.zig").expectEqual;
const minimumBackward = @import("minimum_maximum_backward.zig").backward;
const Context = @import("backward.zig").Context;

fn minimumScalar(comptime T: type) T {
    return switch (T) {
        f64 => std.math.f64_max,
        f32 => std.math.f32_max,
        f16 => std.math.f16_max,
        else => std.math.maxInt(T),
    };
}

pub fn minimum(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), parameters: ReduceParameters) !CpuTensor(T) {
    return try reduce(T, allocator, tensor, struct {
        fn call(accumulator: T, value: T) T {
            return std.math.min(accumulator, value);
        }
    }.call, minimumScalar(T), parameters);
}

test "minimum rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try minimum(f64, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "minimum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 5, 10, 7, 8, 10 });
    const actual = try minimum(i32, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(i32, &arena.allocator, 5);
    expectEqual(i32, actual, expected);
}

test "minimum rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try minimum(f16, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(f16, &arena.allocator, 5);
    expectEqual(f16, actual, expected);
}

test "minimum rank 2 across 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try minimum(f16, &arena.allocator, x, ReduceParameters{ .dimension = 0 });
    const expected = try constant(f16, &arena.allocator, .{ -3, 2 });
    expectEqual(f16, actual, expected);
}

test "minimum rank 3" {
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
    const actual = try minimum(i8, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(i8, &arena.allocator, 2);
    expectEqual(i8, actual, expected);
}

test "minimum rank 3 accross 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try minimum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 0 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
    });
    expectEqual(i64, actual, expected);
}

test "minimum rank 3 accross 1 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try minimum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 1 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ -3, 2 },
        .{ 5, 6 },
    });
    expectEqual(i64, actual, expected);
}

test "minimum rank 3 accross 2 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try minimum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 2 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, -3 },
        .{ 5, 7 },
    });
    expectEqual(i64, actual, expected);
}

test "minimum keep dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try minimum(i64, &arena.allocator, x, ReduceParameters{ .keep_dimensions = true });
    const expected = try constant(i64, &arena.allocator, .{
        .{1},
    });
    expectEqual(i64, actual, expected);
}

test "minimum keep dimensions 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try minimum(i64, &arena.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 0,
    });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
    });
    expectEqual(i64, actual, expected);
}

test "minimum keep dimensions 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try minimum(i64, &arena.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 1,
    });
    const expected = try constant(i64, &arena.allocator, .{
        .{1}, .{4},
    });
    expectEqual(i64, actual, expected);
}

test "minimum backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, 4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try minimum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try minimumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try minimum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try minimumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 1, 0, 0, 0, 0 });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 1 repeated min" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 5, 3, 4, 1 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try minimum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try minimumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0.5, 0, 0, 0, 0.5 });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 1 thrice repeated min" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 1, 1, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try minimum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try minimumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0.3333, 0.3333, 0.3333, 0, 0 });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try minimum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try minimumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 1, 0 },
        .{ 0, 0 },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 3 dimension 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const parameters = ReduceParameters{ .dimension = 0 };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0.25 },
        },
        .{
            .{ 0, 0.25 },
            .{ 0, 0 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 3 dimension 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const parameters = ReduceParameters{ .dimension = 1 };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0, 0 },
            .{ 0.25, 0.25 },
        },
        .{
            .{ 0.25, 0.25 },
            .{ 0, 0 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 3 dimension 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const parameters = ReduceParameters{ .dimension = 2 };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0 },
        },
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward rank 3 dimension 2 repeating min" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 1 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 1 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const parameters = ReduceParameters{ .dimension = 2 };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.125, 0.125 },
            .{ 0.25, 0 },
        },
        .{
            .{ 0.25, 0 },
            .{ 0, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward keep dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const parameters = ReduceParameters{ .keep_dimensions = true };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{.{1}});
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 1, 0, 0 },
        .{ 0, 0, 0 },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum backward keep dimensions 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const parameters = ReduceParameters{
        .dimension = 0,
        .keep_dimensions = true,
    };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{.{ 1. / 3., 1. / 3., 1. / 3. }});
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 1. / 3., 1. / 3., 1. / 3. },
        .{ 0, 0, 0 },
    });
    expectEqual(f64, actual[0], expected);
}

test "minimum bacward keep dimensions 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const parameters = ReduceParameters{
        .dimension = 1,
        .keep_dimensions = true,
    };
    const forward_output = try minimum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{ .{0.5}, .{0.5} });
    const actual = try minimumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0.5, 0, 0 },
        .{ 0.5, 0, 0 },
    });
    expectEqual(f64, actual[0], expected);
}
