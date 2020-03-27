const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const copy = cpu_tensor.copy;
const CpuTensor = cpu_tensor.CpuTensor;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const onesLike = @import("ones_like.zig").onesLike;
const multiply = @import("multiply.zig").multiply;
const sum = @import("sum.zig").sum;
const negate = @import("negate.zig").negate;
const broadcast = @import("broadcast.zig");
const zip = @import("zip.zig").zip;
const debroadcastIndex = broadcast.debroadcastIndex;
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;
const ReduceParameters = @import("reduce.zig").ReduceParameters;

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
    defer c.deinit(allocator);
    const d = try divide(T, allocator, c, b);
    defer d.deinit(allocator);
    return try multiply(T, allocator, d, g);
}

fn outputs1(comptime T: type, context: backward.Context(T)) !CpuTensor(T) {
    const a = context.forward_inputs[0];
    const b = context.forward_inputs[1];
    const g = context.gradient_input;
    const allocator = context.allocator;
    const c = try divide(T, allocator, context.forward_output, b);
    defer c.deinit(allocator);
    const d = try negate(T, allocator, c);
    defer d.deinit(allocator);
    return try multiply(T, allocator, d, g);
}

pub fn divideBackwardBroadcast(comptime T: type, context: backward.Context(T), outputs: []CpuTensor(T)) !void {
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
        const temp = try outputs0(T, context);
        defer temp.deinit(allocator);
        outputs[0] = try sum(T, allocator, temp, ReduceParameters{});
        outputs[1] = try outputs1(T, context);
    } else if (b.shape.len == 0) {
        outputs[0] = try outputs0(T, context);
        const temp = try outputs1(T, context);
        defer temp.deinit(allocator);
        outputs[1] = try sum(T, allocator, temp, ReduceParameters{});
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
    const forward_output = try divide(f64, &arena.allocator, x, y);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
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
    const forward_output = try divide(f64, &arena.allocator, x, y);
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
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
    const forward_output = try divide(f64, &arena.allocator, x, y);
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
        .forward_output = forward_output,
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
    const forward_output = try divide(f64, &arena.allocator, scalar, tensor);
    const forward_output2 = try divide(f64, &arena.allocator, tensor, scalar);
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
        .forward_output = forward_output,
    });
    const actual2 = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ tensor, scalar },
        .forward_output = forward_output2,
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
    const forward_output = try divide(f64, &arena.allocator, rank3, rank4);
    const forward_output2 = try divide(f64, &arena.allocator, rank4, rank3);
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
        .forward_output = forward_output,
    });
    const actual2 = try divideBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ rank4, rank3 },
        .forward_output = forward_output2,
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

test "divide matrix result seperate lifetime" {
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
    const actual = try divide(f64, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 1. / 5., 1. / 3. },
        .{ 3. / 7., 1. / 2. },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "divide matrix broadcast scalar result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const actual = try divide(f64, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 1. / 5., 2. / 5. },
        .{ 3. / 5., 4. / 5. },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "divide scalar broadcast matrix result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const actual = try divide(f64, &leak_allocator.allocator, y, x);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{ 5, 5. / 2. },
        .{ 5. / 3., 5. / 4. },
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "divide broadcast result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{.{ 1, 2 }});
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 1, 2 }},
        .{.{ 3, 4 }},
        .{.{ 5, 6 }},
    });
    const actual = try divide(f64, &leak_allocator.allocator, x, y);
    defer actual.deinit(&leak_allocator.allocator);
    const expected = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 1, 1 }},
        .{.{ 1. / 3., 1. / 2. }},
        .{.{ 1. / 5., 1. / 3. }},
    });
    defer expected.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual, expected);
}

test "gradient divide matrix result seperate lifetime" {
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
    const forward_output = try divide(f64, &leak_allocator.allocator, x, y);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });

    const actual = try divideBackward(f64, backward.Context(f64){
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
        .{ 5.0e-02, 4.1666666666666664e-02 },
        .{ 3.571428571428571e-02, 3.125e-02 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{ -1.0e-02, -1.3888888888888888e-02 },
        .{ -1.5306122448979591e-02, -1.5625e-02 },
    });
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "gradient divide matrix broadcast scalar result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const forward_output = try divide(f64, &leak_allocator.allocator, x, y);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });

    const actual = try divideBackward(f64, backward.Context(f64){
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
        .{ 5.0e-02, 5.0e-02 },
        .{ 5.0e-02, 5.0e-02 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, -0.1);
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "gradient divide scalar broadcast matrix result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &leak_allocator.allocator, 5);
    const forward_output = try divide(f64, &leak_allocator.allocator, y, x);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });

    const actual = try divideBackward(f64, backward.Context(f64){
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
        .{ -1.25e+00, -3.125e-01 },
        .{ -1.388888888888889e-01, -7.8125e-02 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, 5.208333333333333e-01);
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_y_gradient);
    expectEqual(f64, actual[1], expected_x_gradient);
}

test "gradient divide broadcast result seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(f64, &leak_allocator.allocator, .{.{ 1, 2 }});
    const y = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 1, 2 }},
        .{.{ 3, 4 }},
        .{.{ 5, 6 }},
    });
    const forward_output = try divide(f64, &leak_allocator.allocator, x, y);
    const gradient_input = try constant(f64, &leak_allocator.allocator, .{
        .{.{ 0.25, 0.25 }},
        .{.{ 0.25, 0.25 }},
        .{.{ 0.25, 0.25 }},
    });

    const actual = try divideBackward(f64, backward.Context(f64){
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
        .{ 3.833333333333333e-01, 2.2916666666666666e-01 },
    });
    defer expected_x_gradient.deinit(&leak_allocator.allocator);
    const expected_y_gradient = try constant(f64, &leak_allocator.allocator, .{
        .{.{ -2.5e-01, -1.25e-01 }},
        .{.{ -2.7777777777777776e-02, -3.125e-02 }},
        .{.{ -1.0e-02, -1.3888888888888888e-02 }},
    });
    defer expected_y_gradient.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    y.deinit(&leak_allocator.allocator);
    forward_output.deinit(&leak_allocator.allocator);
    gradient_input.deinit(&leak_allocator.allocator);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}
