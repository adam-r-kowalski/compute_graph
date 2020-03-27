const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const reduce = @import("reduce.zig").reduce;
const ReduceParameters = @import("reduce.zig").ReduceParameters;
const expectEqual = @import("../testing.zig").expectEqual;
pub const maximumBackward = @import("minimum_maximum_backward.zig").backward;
const Context = @import("backward.zig").Context;

fn minimumScalar(comptime T: type) T {
    return switch (T) {
        f64 => std.math.f64_min,
        f32 => std.math.f32_min,
        f16 => std.math.f16_min,
        else => std.math.minInt(T),
    };
}

pub fn maximum(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), parameters: ReduceParameters) !CpuTensor(T) {
    return try reduce(T, allocator, tensor, struct {
        fn call(accumulator: T, value: T) T {
            return std.math.max(accumulator, value);
        }
    }.call, minimumScalar(T), parameters);
}

test "maximum rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try maximum(f64, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "maximum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 5, 10, 7, 8, 10 });
    const actual = try maximum(i32, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(i32, &arena.allocator, 10);
    expectEqual(i32, actual, expected);
}

test "maximum rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try maximum(f16, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(f16, &arena.allocator, 10);
    expectEqual(f16, actual, expected);
}

test "maximum rank 2 across 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try maximum(f16, &arena.allocator, x, ReduceParameters{ .dimension = 0 });
    const expected = try constant(f16, &arena.allocator, .{ 5, 6 });
    expectEqual(f16, actual, expected);
}

test "maximum rank 3" {
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
    const actual = try maximum(i8, &arena.allocator, x, ReduceParameters{});
    const expected = try constant(i8, &arena.allocator, 10);
    expectEqual(i8, actual, expected);
}

test "maximum rank 3 accross 0 dimension" {
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
    const actual = try maximum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 0 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    expectEqual(i64, actual, expected);
}

test "maximum rank 3 accross 1 dimension" {
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
    const actual = try maximum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 1 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, 4 },
        .{ 7, 8 },
    });
    expectEqual(i64, actual, expected);
}

test "maximum rank 3 accross 2 dimension" {
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
    const actual = try maximum(i64, &arena.allocator, x, ReduceParameters{ .dimension = 2 });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    expectEqual(i64, actual, expected);
}

test "maximum keep dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try maximum(i64, &arena.allocator, x, ReduceParameters{ .keep_dimensions = true });
    const expected = try constant(i64, &arena.allocator, .{
        .{6},
    });
    expectEqual(i64, actual, expected);
}

test "maximum keep dimensions 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try maximum(i64, &arena.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 0,
    });
    const expected = try constant(i64, &arena.allocator, .{
        .{ 4, 5, 6 },
    });
    expectEqual(i64, actual, expected);
}

test "maximum keep dimensions 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try maximum(i64, &arena.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 1,
    });
    const expected = try constant(i64, &arena.allocator, .{
        .{3}, .{6},
    });
    expectEqual(i64, actual, expected);
}

test "maximum backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, 4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try maximumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try maximumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 0, 0, 0, 1 });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 1 repeated max" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 5, 3, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try maximumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 0.5, 0, 0, 0.5 });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 1 thrice repeated max" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{ 1, 5, 5, 4, 5 });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try maximumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{ 0, 0.3333, 0.3333, 0, 0.3333 });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try maximum(f64, &arena.allocator, forward_input, ReduceParameters{});
    const actual = try maximumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 0 },
        .{ 0, 1 },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 0" {
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
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0, 0.25 },
            .{ 0, 0 },
        },
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 1" {
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
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.25 },
            .{ 0, 0 },
        },
        .{
            .{ 0, 0 },
            .{ 0.25, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 2" {
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
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0, 0.25 },
            .{ 0, 0.25 },
        },
        .{
            .{ 0, 0.25 },
            .{ 0, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 3 dimension 2 repeating max" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{
            .{ 12, 12 },
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
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.125, 0.125 },
            .{ 0, 0.25 },
        },
        .{
            .{ 0, 0.25 },
            .{ 0, 0.25 },
        },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward keep dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(f64, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const parameters = ReduceParameters{ .keep_dimensions = true };
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{.{1}});
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 0, 0 },
        .{ 0, 0, 1 },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum backward keep dimensions 0" {
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
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{.{ 1. / 3., 1. / 3., 1. / 3. }});
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 0, 0 },
        .{ 1. / 3., 1. / 3., 1. / 3. },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum bacward keep dimensions 1" {
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
    const forward_output = try maximum(f64, &arena.allocator, forward_input, parameters);
    const gradient_input = try constant(f64, &arena.allocator, .{ .{0.5}, .{0.5} });
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 0, 0, 0.5 },
        .{ 0, 0, 0.5 },
    });
    expectEqual(f64, actual[0], expected);
}

test "maximum rank 2 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f16, &leak_allocator.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try maximum(f16, &leak_allocator.allocator, x, ReduceParameters{});
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f16, &leak_allocator.allocator, 10);
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    expectEqual(f16, actual, expected);
}

test "maximum rank 2 across 0 dimension seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f16, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try maximum(f16, &leak_allocator.allocator, x, ReduceParameters{ .dimension = 0 });
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f16, &leak_allocator.allocator, .{ 5, 6 });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    expectEqual(f16, actual, expected);
}

test "maximum keep dimensions 0 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(i64, &leak_allocator.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const actual = try maximum(i64, &leak_allocator.allocator, x, ReduceParameters{
        .keep_dimensions = true,
        .dimension = 0,
    });
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(i64, &leak_allocator.allocator, .{
        .{ 4, 5, 6 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    expectEqual(i64, actual, expected);
}

test "maximum backward rank 2 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const forward_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &leak_allocator.allocator, 1);
    const forward_output = try maximum(f64, &leak_allocator.allocator, forward_input, ReduceParameters{});
    const actual = try maximumBackward(f64, ReduceParameters{}, Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 0, 0 },
        .{ 0, 1 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    forward_input.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 2 dimension 0 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const forward_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{ 0.5, 0.5 });
    const parameters = ReduceParameters{ .dimension = 0 };
    const forward_output = try maximum(f64, &leak_allocator.allocator, forward_input, parameters);
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 0, 0 },
        .{ 0.5, 0.5 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    forward_input.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}

test "maximum backward rank 2 dimension 0 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const forward_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{.{ 0.5, 0.5 }});
    const parameters = ReduceParameters{
        .dimension = 0,
        .keep_dimensions = true,
    };
    const forward_output = try maximum(f64, &leak_allocator.allocator, forward_input, parameters);
    const actual = try maximumBackward(f64, parameters, Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 0, 0 },
        .{ 0.5, 0.5 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    forward_input.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected);
}
