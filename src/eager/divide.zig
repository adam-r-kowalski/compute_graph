const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const onesLike = @import("ones_like.zig").onesLike;
const multiply = @import("multiply.zig").multiply;
const sum = @import("sum.zig").sum;
const negate = @import("negate.zig").negate;
const broadcast = @import("broadcast.zig");
const zip = broadcast.zip;
const debroadcastIndex = broadcast.debroadcastIndex;
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;

fn divideScalar(comptime T: type, x: T, y: T) T {
    return switch (T) {
        i64, i32, i8 => @divFloor(x, y),
        else => x / y,
    };
}

pub fn divide(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    return try zip(T, allocator, x, y, struct {
        fn call(a: T, b: T) T {
            return divideScalar(T, a, b);
        }
    }.call);
}

test "divide rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    const y = try constant(f64, &arena.allocator, 10);
    const actual = try divide(f64, &arena.allocator, x, y);
    const expected = try constant(f64, &arena.allocator, 0.5);
    expectEqual(f64, actual, expected);
}

test "divide rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const y = try constant(i32, &arena.allocator, .{ 6, -5, 4, -3, -2, 1 });
    const actual = try divide(i32, &arena.allocator, x, y);
    const expected = try constant(i32, &arena.allocator, .{ 0, 0, 0, 1, 2, 6 });
    expectEqual(i32, actual, expected);
}

test "divide rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(f16, &arena.allocator, .{
        .{ 6, -5 },
        .{ 4, -3 },
        .{ -2, 1 },
    });
    const actual = try divide(f16, &arena.allocator, x, y);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 0.1666, 0.4 },
        .{ 0.75, 1.3333 },
        .{ 2.5, 6 },
    });
    expectEqual(f16, actual, expected);
}

test "divide rank 3" {
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
            .{ 8, -7 },
            .{ 6, -5 },
        },
        .{
            .{ 4, -3 },
            .{ 2, -1 },
        },
    });
    const actual = try divide(i8, &arena.allocator, x, y);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 0, 0 },
            .{ 0, 0 },
        },
        .{
            .{ 1, 2 },
            .{ 3, 8 },
        },
    });
    expectEqual(i8, actual, expected);
}

fn outputs0(comptime T: type, context: backward.Context(T)) !CpuTensor(T) {
    const b = context.forward_inputs[1];
    const g = context.gradient_input;
    const allocator = context.allocator;
    const c = try onesLike(T, allocator, b);
    const d = try divide(T, allocator, c, b);
    return try multiply(T, allocator, d, g);
}

fn outputs1(comptime T: type, context: backward.Context(T)) !CpuTensor(T) {
    const a = context.forward_inputs[0];
    const b = context.forward_inputs[1];
    const g = context.gradient_input;
    const allocator = context.allocator;
    const c = try divide(T, allocator, a, b);
    const d = try divide(T, allocator, c, b);
    const e = try negate(T, allocator, d);
    return try multiply(T, allocator, e, g);
}

pub fn divideBackwardBroadcast(comptime T: type, context: backward.Context(T), outputs: []CpuTensor(T)) !void {
    const allocator = context.allocator;
    const gradient_input = context.gradient_input;
    const gradient_shape = gradient_input.shape;
    const gradient_stride = gradient_input.stride;
    const gradient_array = gradient_input.storage.array;
    const gradient_cartesian_index = try allocator.alloc(usize, gradient_shape.len);
    errdefer allocator.free(gradient_cartesian_index);
    for (gradient_cartesian_index) |*e| e.* = 0;
    const x_shape = context.forward_inputs[0].shape;
    const x_stride = context.forward_inputs[0].stride;
    const x_array = try allocator.alloc(T, context.forward_inputs[0].storage.array.len);
    errdefer allocator.free(x_array);
    for (x_array) |*e| e.* = 0;
    const x_cartesian_index = try allocator.alloc(usize, x_shape.len);
    errdefer allocator.free(x_cartesian_index);
    const y_shape = context.forward_inputs[1].shape;
    const y_stride = context.forward_inputs[1].stride;
    const y_array = try allocator.alloc(T, context.forward_inputs[1].storage.array.len);
    errdefer allocator.free(y_array);
    for (y_array) |*e| e.* = 0;
    const y_cartesian_index = try allocator.alloc(usize, y_shape.len);
    errdefer allocator.free(y_cartesian_index);
    const x_forward_array = context.forward_inputs[0].storage.array;
    const y_forward_array = context.forward_inputs[1].storage.array;
    while (true) {
        debroadcastIndex(x_shape, gradient_cartesian_index, x_cartesian_index);
        debroadcastIndex(y_shape, gradient_cartesian_index, y_cartesian_index);
        const x_index = linearIndex(x_stride, x_cartesian_index);
        const y_index = linearIndex(y_stride, y_cartesian_index);
        const gradient_index = linearIndex(gradient_stride, gradient_cartesian_index);
        const y_value = y_forward_array[y_index];
        const g_value = gradient_array[gradient_index];
        x_array[x_index] += 1 / y_value * g_value;
        y_array[y_index] += -(x_forward_array[x_index] / y_value / y_value) * g_value;
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

pub fn divideBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const outputs = try context.allocator.alloc(CpuTensor(T), 2);
    errdefer context.allocator.free(outputs);
    const a = context.forward_inputs[0];
    const b = context.forward_inputs[1];
    const g = context.gradient_input;
    const allocator = context.allocator;
    if (std.mem.eql(usize, a.shape, b.shape)) {
        outputs[0] = try outputs0(T, context);
        outputs[1] = try outputs1(T, context);
    } else if (a.shape.len == 0) {
        outputs[0] = try sum(T, allocator, try outputs0(T, context), null);
        outputs[1] = try outputs1(T, context);
    } else if (b.shape.len == 0) {
        outputs[0] = try outputs0(T, context);
        outputs[1] = try sum(T, allocator, try outputs1(T, context), null);
    } else {
        try divideBackwardBroadcast(T, context, outputs);
    }
    return outputs;
}

test "divide backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 4);
    const y = try constant(f64, &arena.allocator, 10);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, 0.1);
    const expected_y_gradient = try constant(f64, &arena.allocator, -0.04);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "divide backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const y = try constant(f64, &arena.allocator, .{ 6, 7, 8, 9, 10 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, .{ 0.3333, 0.5714, 0.75, 0.8888, 1.0 });
    const expected_y_gradient = try constant(f64, &arena.allocator, .{ -0.0555, -0.1632, -0.2812, -0.3950, -0.50 });
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "divide backward rank 2" {
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
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, .{
        .{ 0.4, 0.6666 },
        .{ 0.8571, 1 },
    });
    const expected_y_gradient = try constant(f64, &arena.allocator, .{
        .{ -0.080, -0.2222 },
        .{ -0.3673, -0.5 },
    });
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "divide backwards broadcast scalar rank 3" {
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
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ scalar, tensor },
    });
    const actual2 = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ tensor, scalar },
    });
    const expected_scalar_gradient = try constant(f64, &arena.allocator, 0.0793);
    const expected_tensor_gradient = try constant(f64, &arena.allocator, .{
        .{
            .{ 0.625, 0.1562 },
            .{ 0.0694, 0.0391 },
        },
        .{
            .{ 0.025, 0.0174 },
            .{ 0.0128, 0.0098 },
        },
    });
    const expected2_scalar_gradient = try constant(f64, &arena.allocator, 0.02);
    const expected2_tensor_gradient = try constant(f64, &arena.allocator, .{
        .{
            .{ -0.025, -0.025 },
            .{ -0.025, -0.025 },
        },
        .{
            .{ -0.025, -0.025 },
            .{ -0.025, -0.025 },
        },
    });
    expectEqual(f64, actual[0], expected_scalar_gradient);
    expectEqual(f64, actual[1], expected_tensor_gradient);
    expectEqual(f64, actual2[0], expected2_tensor_gradient);
    expectEqual(f64, actual2[1], expected2_scalar_gradient);
}

test "divide backwards broadcast rank 3 to rank 4" {
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
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ rank3, rank4 },
    });
    const actual2 = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ rank4, rank3 },
    });
    const expected_rank_3_gradient = try constant(f64, &arena.allocator, .{
        .{.{ 0.0522, 0.0340 }},
        .{.{ 0.0522, 0.0340 }},
        .{.{ 0.0522, 0.0340 }},
    });
    const expected_rank_4_gradient = try constant(f64, &arena.allocator, .{
        .{.{
            .{ -0.25, -0.0833 },
            .{ -0.0278, -0.0208 },
            .{ -0.01, -0.0093 },
        }},
        .{.{
            .{ -0.0051, -0.0052 },
            .{ -0.0031, -0.0033 },
            .{ -0.0021, -0.0023 },
        }},
    });
    const expected2_rank_3_gradient = try constant(f64, &arena.allocator, .{
        .{.{ -1.0, -0.2917 }},
        .{.{ -0.1111, -0.0729 }},
        .{.{ -0.04, -0.0324 }},
    });
    const expected2_rank_4_gradient = try constant(f64, &arena.allocator, .{
        .{.{
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
        }},
        .{.{
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
        }},
    });
    expectEqual(f64, expected_rank_3_gradient, actual[0]);
    expectEqual(f64, expected_rank_4_gradient, actual[1]);
    expectEqual(f64, expected2_rank_4_gradient, actual2[0]);
    expectEqual(f64, expected2_rank_3_gradient, actual2[1]);
}
