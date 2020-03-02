const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const broadcast = @import("broadcast.zig");
const broadcastShape = broadcast.broadcastShape;
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;
const debroadcastIndex = broadcast.debroadcastIndex;
const sum = @import("sum.zig").sum;

fn addSameShape(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    const shape = x.shape;
    const stride = x.stride;
    switch (x.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = scalar + y.storage.scalar },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            const y_array = y.storage.array;
            for (array) |e, i| new_array[i] = e + y_array[i];
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

fn addBroadcastScalar(comptime T: type, allocator: *Allocator, scalar: T, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    const array = tensor.storage.array;
    const new_array = try allocator.alloc(T, array.len);
    errdefer allocator.free(new_array);
    for (array) |e, i| new_array[i] = scalar + array[i];
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = new_array },
    };
}

fn addBroadcast(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    const shape = try broadcastShape(allocator, x.shape, y.shape);
    errdefer allocator.free(shape);
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    const cartesian_index = try allocator.alloc(usize, shape.len);
    errdefer allocator.free(cartesian_index);
    for (cartesian_index) |*e| e.* = 0;
    const x_cartesian_index = try allocator.alloc(usize, x.shape.len);
    errdefer allocator.free(x_cartesian_index);
    const y_cartesian_index = try allocator.alloc(usize, y.shape.len);
    errdefer allocator.free(y_cartesian_index);
    const array = try allocator.alloc(T, tensorLength(shape));
    errdefer allocator.free(array);
    const x_array = x.storage.array;
    const y_array = y.storage.array;
    while (true) {
        debroadcastIndex(x.shape, cartesian_index, x_cartesian_index);
        debroadcastIndex(y.shape, cartesian_index, y_cartesian_index);
        const x_index = linearIndex(x.stride, x_cartesian_index);
        const y_index = linearIndex(y.stride, y_cartesian_index);
        const index = linearIndex(stride, cartesian_index);
        array[index] = x_array[x_index] + y_array[y_index];
        if (maximumCartesianIndex(shape, cartesian_index)) break;
        incrementCartesianIndex(shape, cartesian_index);
    }
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = array },
    };
}

pub fn add(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    if (std.mem.eql(usize, x.shape, y.shape))
        return try addSameShape(T, allocator, x, y);
    if (x.shape.len == 0)
        return try addBroadcastScalar(T, allocator, x.storage.scalar, y);
    if (y.shape.len == 0)
        return try addBroadcastScalar(T, allocator, y.storage.scalar, x);
    return try addBroadcast(T, allocator, x, y);
}

test "add rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    const y = try constant(f64, &arena.allocator, 10);
    const actual = try add(f64, &arena.allocator, x, y);
    const expected = try constant(f64, &arena.allocator, 15);
    expectEqual(f64, actual, expected);
}

test "add rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try add(i32, &arena.allocator, x, x);
    const expected = try constant(i32, &arena.allocator, .{ 2, -4, 6, -8, -10, 12 });
    expectEqual(i32, actual, expected);
}

test "add rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try add(f16, &arena.allocator, x, x);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f16, actual, expected);
}

test "add rank 3" {
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
    const actual = try add(i8, &arena.allocator, x, x);
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

test "add broadcast scalar rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(i32, &arena.allocator, 5);
    const tensor = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try add(i32, &arena.allocator, scalar, tensor);
    const actual2 = try add(i32, &arena.allocator, tensor, scalar);
    const expected = try constant(i32, &arena.allocator, .{ 6, 3, 8, 1, 0, 11 });
    expectEqual(i32, actual, expected);
    expectEqual(i32, actual2, expected);
}

test "add broadcast scalar rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(f16, &arena.allocator, 3);
    const tensor = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try add(f16, &arena.allocator, scalar, tensor);
    const actual2 = try add(f16, &arena.allocator, tensor, scalar);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 4, 1 },
        .{ 6, -1 },
        .{ -2, 9 },
    });
    expectEqual(f16, actual, expected);
    expectEqual(f16, actual2, expected);
}

test "add broadcast scalar rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(i8, &arena.allocator, -5);
    const tensor = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const actual = try add(i8, &arena.allocator, scalar, tensor);
    const actual2 = try add(i8, &arena.allocator, tensor, scalar);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ -4, -7 },
            .{ -2, -9 },
        },
        .{
            .{ 0, -11 },
            .{ 2, -13 },
        },
    });
    expectEqual(i8, actual, expected);
    expectEqual(i8, actual2, expected);
}

test "add broadcast rank 3 to rank 1" {
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
    const actual = try add(i8, &arena.allocator, rank3, rank1);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, 3, 5 },
            .{ 4, 6, 8 },
        },
        .{
            .{ 7, 9, 11 },
            .{ 10, 12, 14 },
        },
        .{
            .{ 13, 15, 17 },
            .{ 16, 18, 20 },
        },
    });
    expectEqual(i8, actual, expected);
}

test "add broadcast rank 3 to rank 4" {
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

    const actual = try add(i64, &arena.allocator, rank3, rank4);
    const expected = try constant(i64, &arena.allocator, .{
        .{
            .{
                .{ 2, 4 },
                .{ 4, 6 },
                .{ 6, 8 },
            },
            .{
                .{ 4, 6 },
                .{ 6, 8 },
                .{ 8, 10 },
            },
            .{
                .{ 6, 8 },
                .{ 8, 10 },
                .{ 10, 12 },
            },
        },
        .{
            .{
                .{ 8, 10 },
                .{ 10, 12 },
                .{ 12, 14 },
            },
            .{
                .{ 10, 12 },
                .{ 12, 14 },
                .{ 14, 16 },
            },
            .{
                .{ 12, 14 },
                .{ 14, 16 },
                .{ 16, 18 },
            },
        },
    });
    expectEqual(i64, actual, expected);
}

pub fn addBackwardBroadcast(comptime T: type, context: backward.Context(T), outputs: []CpuTensor(T)) !void {
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
    while (true) {
        debroadcastIndex(x_shape, gradient_cartesian_index, x_cartesian_index);
        debroadcastIndex(y_shape, gradient_cartesian_index, y_cartesian_index);
        const x_index = linearIndex(x_stride, x_cartesian_index);
        const y_index = linearIndex(y_stride, y_cartesian_index);
        const gradient_index = linearIndex(gradient_stride, gradient_cartesian_index);
        x_array[x_index] += gradient_array[gradient_index];
        y_array[y_index] += gradient_array[gradient_index];
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

pub fn addBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const outputs = try context.allocator.alloc(CpuTensor(T), 2);
    errdefer context.allocator.free(outputs);
    const inputs = context.forward_inputs;
    if (std.mem.eql(usize, inputs[0].shape, inputs[1].shape)) {
        outputs[0] = context.gradient_input;
        outputs[1] = context.gradient_input;
    } else if (inputs[0].shape.len == 0) {
        outputs[0] = sum(T, context.allocator, context.gradient_input, null) catch unreachable;
        outputs[1] = context.gradient_input;
    } else if (inputs[1].shape.len == 0) {
        outputs[0] = context.gradient_input;
        outputs[1] = sum(T, context.allocator, context.gradient_input, null) catch unreachable;
    } else {
        try addBackwardBroadcast(T, context, outputs);
    }
    return outputs;
}

test "add backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 4);
    const y = try constant(f64, &arena.allocator, 10);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const y = try constant(f64, &arena.allocator, .{ 6, 7, 8, 9, 10 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backward rank 2" {
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
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backwards broadcast scalar rank 3" {
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
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ scalar, tensor },
    });
    const actual2 = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ tensor, scalar },
    });
    const expected_scalar_gradient = try constant(f64, &arena.allocator, 1);
    const expected_tensor_gradient = try constant(f64, &arena.allocator, .{
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
    expectEqual(f64, actual[0], expected_scalar_gradient);
    expectEqual(f64, actual[1], expected_tensor_gradient);
    expectEqual(f64, actual2[0], expected_tensor_gradient);
    expectEqual(f64, actual2[1], expected_scalar_gradient);
}

test "add backwards broadcast rank 3 to rank 4" {
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
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ rank3, rank4 },
    });
    const expected_rank_3_gradient = try constant(f64, &arena.allocator, .{
        .{.{ 1.6666666666666669e-01, 1.6666666666666669e-01 }},
        .{.{ 1.6666666666666669e-01, 1.6666666666666669e-01 }},
        .{.{ 1.6666666666666669e-01, 1.6666666666666669e-01 }},
    });
    const expected_rank_4_gradient = try constant(f64, &arena.allocator, .{
        .{.{
            .{ 8.333333333333333e-02, 8.333333333333333e-02 },
            .{ 8.333333333333333e-02, 8.333333333333333e-02 },
            .{ 8.333333333333333e-02, 8.333333333333333e-02 },
        }},
        .{.{
            .{ 8.333333333333333e-02, 8.333333333333333e-02 },
            .{ 8.333333333333333e-02, 8.333333333333333e-02 },
            .{ 8.333333333333333e-02, 8.333333333333333e-02 },
        }},
    });
    expectEqual(f64, expected_rank_3_gradient, actual[0]);
    expectEqual(f64, expected_rank_4_gradient, actual[1]);
}
