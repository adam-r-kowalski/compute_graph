const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const copy = cpu_tensor.copy;
const CpuTensor = cpu_tensor.CpuTensor;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const zip = @import("zip.zig").zip;
const sum = @import("sum.zig").sum;
const broadcast = @import("broadcast.zig");
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;
const debroadcastIndex = broadcast.debroadcastIndex;
const map = @import("map.zig").map;
const ReduceParameters = @import("reduce.zig").ReduceParameters;

pub fn subtract(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    return try zip(T, allocator, x, y, struct {
        fn call(a: T, b: T) T {
            return a - b;
        }
    }.call);
}

test "subtract rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    const y = try constant(f64, &arena.allocator, 10);
    const actual = try subtract(f64, &arena.allocator, x, y);
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "subtract rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const y = try constant(i32, &arena.allocator, .{ -1, 2, -3, 4, 5, -6 });
    const actual = try subtract(i32, &arena.allocator, x, y);
    const expected = try constant(i32, &arena.allocator, .{ 2, -4, 6, -8, -10, 12 });
    expectEqual(i32, actual, expected);
}

test "subtract rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(f16, &arena.allocator, .{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    const actual = try subtract(f16, &arena.allocator, x, y);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f16, actual, expected);
}

test "subtract rank 3" {
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
    const y = try constant(i8, &arena.allocator, .{
        .{
            .{ -1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ -5, 6 },
            .{ -7, 8 },
        },
    });
    const actual = try subtract(i8, &arena.allocator, x, y);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 2, -4 },
            .{ 6, -8 },
        },
        .{
            .{ 10, -12 },
            .{ 14, -16 },
        },
    });
    expectEqual(i8, actual, expected);
}

test "subtract broadcast scalar rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(i32, &arena.allocator, 5);
    const tensor = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try subtract(i32, &arena.allocator, scalar, tensor);
    const actual2 = try subtract(i32, &arena.allocator, tensor, scalar);
    const expected = try constant(i32, &arena.allocator, .{ 4, 7, 2, 9, 10, -1 });
    const expected2 = try constant(i32, &arena.allocator, .{ -4, -7, -2, -9, -10, 1 });
    expectEqual(i32, actual, expected);
    expectEqual(i32, actual2, expected2);
}

test "subtract broadcast scalar rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(f16, &arena.allocator, 3);
    const tensor = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try subtract(f16, &arena.allocator, scalar, tensor);
    const actual2 = try subtract(f16, &arena.allocator, tensor, scalar);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 2, 5 },
        .{ 0, 7 },
        .{ 8, -3 },
    });
    const expected2 = try constant(f16, &arena.allocator, .{
        .{ -2, -5 },
        .{ 0, -7 },
        .{ -8, 3 },
    });
    expectEqual(f16, actual, expected);
    expectEqual(f16, actual2, expected2);
}

test "subtract broadcast rank 3 to rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rank3 = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, 2, 3 },
            .{ 4, 5, 6 },
        },
        .{
            .{ 7, 8, 9 },
            .{ 10, 11, 12 },
        },
        .{
            .{ 13, 14, 15 },
            .{ 16, 17, 18 },
        },
    });
    const rank1 = try constant(i8, &arena.allocator, .{ 0, 1, 2 });
    const actual = try subtract(i8, &arena.allocator, rank3, rank1);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, 1, 1 },
            .{ 4, 4, 4 },
        },
        .{
            .{ 7, 7, 7 },
            .{ 10, 10, 10 },
        },
        .{
            .{ 13, 13, 13 },
            .{ 16, 16, 16 },
        },
    });
    expectEqual(i8, actual, expected);
}

test "subtract broadcast rank 3 to rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rank3 = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
        },
        .{
            .{ 3, 4 },
        },
        .{
            .{ 5, 6 },
        },
    });

    const rank4 = try constant(i64, &arena.allocator, .{
        .{.{
            .{ 1, 2 },
            .{ 3, 4 },
            .{ 5, 6 },
        }},
        .{.{
            .{ 7, 8 },
            .{ 9, 10 },
            .{ 11, 12 },
        }},
    });

    const actual = try subtract(i64, &arena.allocator, rank3, rank4);
    const expected = try constant(i64, &arena.allocator, .{
        .{
            .{
                .{ 0, 0 },
                .{ -2, -2 },
                .{ -4, -4 },
            },
            .{
                .{ 2, 2 },
                .{ 0, 0 },
                .{ -2, -2 },
            },
            .{
                .{ 4, 4 },
                .{ 2, 2 },
                .{ 0, 0 },
            },
        },
        .{
            .{
                .{ -6, -6 },
                .{ -8, -8 },
                .{ -10, -10 },
            },
            .{
                .{ -4, -4 },
                .{ -6, -6 },
                .{ -8, -8 },
            },
            .{
                .{ -2, -2 },
                .{ -4, -4 },
                .{ -6, -6 },
            },
        },
    });
    expectEqual(i64, actual, expected);
}

// TODO(refactor) unify with subtract backward broadcast
pub fn subtractBackwardBroadcast(comptime T: type, context: backward.Context(T), outputs: []CpuTensor(T)) !void {
    const allocator = context.allocator;
    const gradient_input = context.gradient_input;
    const gradient_shape = gradient_input.shape;
    const gradient_stride = gradient_input.stride;
    const gradient_array = gradient_input.storage.array;
    const gradient_cartesian_index = try allocator.alloc(usize, gradient_shape.len);
    defer allocator.free(gradient_cartesian_index);
    for (gradient_cartesian_index) |*e| e.* = 0;

    const x_shape = try copy(usize, allocator, context.forward_inputs[0].shape);
    errdefer allocator.free(x_shape);
    const x_stride = try copy(usize, allocator, context.forward_inputs[0].stride);
    errdefer allocator.free(x_stride);
    const x_array = try allocator.alloc(T, context.forward_inputs[0].storage.array.len);
    errdefer allocator.free(x_array);
    for (x_array) |*e| e.* = 0;
    const x_cartesian_index = try allocator.alloc(usize, x_shape.len);
    defer allocator.free(x_cartesian_index);

    const y_shape = try copy(usize, allocator, context.forward_inputs[1].shape);
    errdefer allocator.free(y_shape);
    const y_stride = try copy(usize, allocator, context.forward_inputs[1].stride);
    errdefer allocator.free(y_stride);
    const y_array = try allocator.alloc(T, context.forward_inputs[1].storage.array.len);
    errdefer allocator.free(y_array);
    for (y_array) |*e| e.* = 0;
    const y_cartesian_index = try allocator.alloc(usize, y_shape.len);
    defer allocator.free(y_cartesian_index);

    while (true) {
        debroadcastIndex(x_shape, gradient_cartesian_index, x_cartesian_index);
        debroadcastIndex(y_shape, gradient_cartesian_index, y_cartesian_index);
        const x_index = linearIndex(x_stride, x_cartesian_index);
        const y_index = linearIndex(y_stride, y_cartesian_index);
        const gradient_index = linearIndex(gradient_stride, gradient_cartesian_index);
        x_array[x_index] += gradient_array[gradient_index];
        y_array[y_index] -= gradient_array[gradient_index];
        if (maximumCartesianIndex(gradient_shape, gradient_cartesian_index)) break;
        incrementCartesianIndex(gradient_shape, gradient_cartesian_index);
    }
    outputs[0] = CpuTensor(T){
        .shape = x_shape,
        .stride = x_stride,
        .storage = .{ .array = x_array },
    };
    outputs[1] = CpuTensor(T){
        .shape = y_shape,
        .stride = y_stride,
        .storage = .{ .array = y_array },
    };
}

pub fn subtractBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const allocator = context.allocator;
    const outputs = try allocator.alloc(CpuTensor(T), 2);
    errdefer allocator.free(outputs);
    const inputs = context.forward_inputs;
    const negate = struct {
        fn call(t: T) T {
            return -1 * t;
        }
    }.call;
    if (std.mem.eql(usize, inputs[0].shape, inputs[1].shape)) {
        outputs[0] = try context.gradient_input.copy(allocator);
        outputs[1] = try map(T, allocator, context.gradient_input, negate);
    } else if (inputs[0].shape.len == 0) {
        outputs[0] = try sum(T, allocator, context.gradient_input, ReduceParameters{});
        outputs[1] = try map(T, allocator, context.gradient_input, negate);
    } else if (inputs[1].shape.len == 0) {
        outputs[0] = try context.gradient_input.copy(allocator);
        // TODO(performance) fuse scale and sum into single operation using map reduce
        const scaled = try map(T, allocator, context.gradient_input, negate);
        defer scaled.deinit(allocator);
        outputs[1] = try sum(T, allocator, scaled, ReduceParameters{});
    } else {
        try subtractBackwardBroadcast(T, context, outputs);
    }
    return outputs;
}

test "subtract backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 4);
    const y = try constant(f64, &arena.allocator, 10);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const forward_output = try subtract(f64, &arena.allocator, x, y);
    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    const expected_a_gradient = try constant(f64, &arena.allocator, 1);
    const expected_b_gradient = try constant(f64, &arena.allocator, -1);
    expectEqual(f64, actual[0], expected_a_gradient);
    expectEqual(f64, actual[1], expected_b_gradient);
}

test "subtract backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const y = try constant(f64, &arena.allocator, .{ 6, 7, 8, 9, 10 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const forward_output = try subtract(f64, &arena.allocator, x, y);
    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    const expected_a_gradient = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const expected_b_gradient = try constant(f64, &arena.allocator, .{ -2, -4, -6, -8, -10 });
    expectEqual(f64, actual[0], expected_a_gradient);
    expectEqual(f64, actual[1], expected_b_gradient);
}

test "subtract backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &arena.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const forward_output = try subtract(f64, &arena.allocator, x, y);
    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    const expected_a_gradient = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const expected_b_gradient = try constant(f64, &arena.allocator, .{
        .{ -2, -4 },
        .{ -6, -8 },
    });
    expectEqual(f64, actual[0], expected_a_gradient);
    expectEqual(f64, actual[1], expected_b_gradient);
}

test "subtract backwards broadcast scalar rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(f64, &arena.allocator, -5);
    const tensor = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{
            .{
                0.125,
                0.125,
            },
            .{
                0.125,
                0.125,
            },
        },
        .{
            .{
                0.125,
                0.125,
            },
            .{
                0.125,
                0.125,
            },
        },
    });
    const forward_output = try subtract(f64, &arena.allocator, scalar, tensor);
    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ scalar, tensor },
        .forward_output = forward_output,
    });
    const expected_scalar_gradient = try constant(f64, &arena.allocator, 1);
    const expected_tensor_gradient = try constant(f64, &arena.allocator, .{
        .{
            .{ -0.125, -0.125 },
            .{ -0.125, -0.125 },
        },
        .{
            .{ -0.125, -0.125 },
            .{ -0.125, -0.125 },
        },
    });
    expectEqual(f64, expected_scalar_gradient, actual[0]);
    expectEqual(f64, expected_tensor_gradient, actual[1]);
}

test "subtract backwards broadcast rank 3 scalar " {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(f64, &arena.allocator, -5);
    const tensor = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{
            .{
                0.125,
                0.125,
            },
            .{
                0.125,
                0.125,
            },
        },
        .{
            .{
                0.125,
                0.125,
            },
            .{
                0.125,
                0.125,
            },
        },
    });
    const forward_output = try subtract(f64, &arena.allocator, tensor, scalar);
    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ tensor, scalar },
        .forward_output = forward_output,
    });
    const expected_scalar_gradient = try constant(f64, &arena.allocator, -1);
    const expected_tensor_gradient = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.125, 0.125 },
            .{ 0.125, 0.125 },
        },
        .{
            .{ 0.125, 0.125 },
            .{ 0.125, 0.125 },
        },
    });
    expectEqual(f64, expected_scalar_gradient, actual[1]);
    expectEqual(f64, expected_tensor_gradient, actual[0]);
}

test "subtract backwards broadcast rank 3 to rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rank3 = try constant(f64, &arena.allocator, .{
        .{
            .{ 1, 2 },
        },
        .{
            .{ 3, 4 },
        },
        .{
            .{ 5, 6 },
        },
    });

    const rank4 = try constant(f64, &arena.allocator, .{
        .{.{
            .{ 1, 2 },
            .{ 3, 4 },
            .{ 5, 6 },
        }},
        .{.{
            .{ 7, 8 },
            .{ 9, 10 },
            .{ 11, 12 },
        }},
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{
            .{
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
            },
            .{
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
            },
            .{
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
            },
        },
        .{
            .{
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
            },
            .{
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
            },
            .{
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
                .{ 1. / 36., 1. / 36. },
            },
        },
    });
    const forward_output = try subtract(f64, &arena.allocator, rank3, rank4);
    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ rank3, rank4 },
        .forward_output = forward_output,
    });
    const expected_rank_3_gradient = try constant(f64, &arena.allocator, .{
        .{.{ 0.16667, 0.16667 }},
        .{.{ 0.16667, 0.16667 }},
        .{.{ 0.16667, 0.16667 }},
    });
    const expected_rank_4_gradient = try constant(f64, &arena.allocator, .{
        .{.{
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
        }},
        .{.{
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
        }},
    });
    expectEqual(f64, expected_rank_3_gradient, actual[0]);
    expectEqual(f64, expected_rank_4_gradient, actual[1]);
}

test "subtract matrix result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const actual = try subtract(f64, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ -4, -4 },
        .{ -4, -4 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "subtract matrix broadcast scalar result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const actual = try subtract(f64, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ -4, -3 },
        .{ -2, -1 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "subtract scalar broadcast matrix result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const actual = try subtract(f64, &leak_allocator.allocator, y, x);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 4, 3 },
        .{ 2, 1 },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "subtract broadcast result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{.{ 1, 2 }});
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 1, 2 }},
        .{.{ 3, 4 }},
        .{.{ 5, 6 }},
    });
    const actual = try subtract(f64, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 0, 0 }},
        .{.{ -2, -2 }},
        .{.{ -4, -4 }},
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "gradient subtract matrix result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const forward_output = try subtract(f64, &leak_allocator.allocator, x, y);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });

    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected_x_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ -0.25, -0.25 },
        .{ -0.25, -0.25 },
    });
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "gradient subtract matrix broadcast scalar result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const forward_output = try subtract(f64, &leak_allocator.allocator, x, y);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });

    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected_x_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, -1);
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "gradient subtract scalar broadcast matrix result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const forward_output = try subtract(f64, &leak_allocator.allocator, y, x);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });

    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ y, x },
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected_x_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ -0.25, -0.25 },
        .{ -0.25, -0.25 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, 1);
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_y_gradient);
    expectEqual(f64, actual[1], expected_x_gradient);
}

test "gradient subtract broadcast result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{.{ 1, 2 }});
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 1, 2 }},
        .{.{ 3, 4 }},
        .{.{ 5, 6 }},
    });
    const forward_output = try subtract(f64, &leak_allocator.allocator, x, y);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 0.25, 0.25 }},
        .{.{ 0.25, 0.25 }},
        .{.{ 0.25, 0.25 }},
    });

    const actual = try subtractBackward(f64, backward.Context(f64){
        .allocator = &leak_allocator.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
    });
    defer {
        for (actual) |tensor| tensor.deinit(&leak_allocator.allocator);
        leak_allocator.allocator.free(actual);
    }
    const expected_x_gradient = try constant(f64, &leak_allocator.allocator, .{.{ 7.5e-01, 7.5e-01 }});
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{.{ -2.5e-01, -2.5e-01 }},
        .{.{ -2.5e-01, -2.5e-01 }},
        .{.{ -2.5e-01, -2.5e-01 }},
    });
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}
