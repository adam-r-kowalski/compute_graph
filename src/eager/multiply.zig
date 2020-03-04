const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const zip = @import("broadcast.zig").zip;
const map = @import("map.zig").map;
const sum = @import("sum.zig").sum;

pub fn multiply(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    return try zip(T, allocator, x, y, struct {
        fn call(a: T, b: T) T {
            return a * b;
        }
    }.call);
}

test "multiply rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    const y = try constant(f64, &arena.allocator, 10);
    const actual = try multiply(f64, &arena.allocator, x, y);
    const expected = try constant(f64, &arena.allocator, 50);
    expectEqual(f64, actual, expected);
}

test "multiply rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try multiply(i32, &arena.allocator, x, x);
    const expected = try constant(i32, &arena.allocator, .{ 1, 4, 9, 16, 25, 36 });
    expectEqual(i32, actual, expected);
}

test "multiply rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try multiply(f16, &arena.allocator, x, x);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 1, 4 },
        .{ 9, 16 },
        .{ 25, 36 },
    });
    expectEqual(f16, actual, expected);
}

test "multiply rank 3" {
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
    const actual = try multiply(i8, &arena.allocator, x, x);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, 4 },
            .{ 9, 16 },
        },
        .{
            .{ 25, 36 },
            .{ 49, 64 },
        },
    });
    expectEqual(i8, actual, expected);
}

pub fn multiplyBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const outputs = try context.allocator.alloc(CpuTensor(T), 2);
    errdefer context.allocator.free(outputs);
    const inputs = context.forward_inputs;
    const Closure = struct {
        scalar: T,
        pub fn call(self: @This(), t: T) T {
            return t * self.scalar;
        }
    };
    if (std.mem.eql(usize, inputs[0].shape, inputs[1].shape)) {
        outputs[0] = try multiply(T, context.allocator, context.gradient_input, inputs[1]);
        outputs[1] = try multiply(T, context.allocator, context.gradient_input, inputs[0]);
    } else if (inputs[0].shape.len == 0) {
        // TODO(performance) fuse multiply and sum into single operation using map reduce
        const multiplied = try multiply(T, context.allocator, context.gradient_input, inputs[1]);
        outputs[0] = try sum(T, context.allocator, multiplied, null);
        outputs[1] = try map(T, context.allocator, context.gradient_input, Closure{ .scalar = inputs[0].storage.scalar });
    } else if (inputs[1].shape.len == 0) {
        outputs[0] = try map(T, context.allocator, context.gradient_input, Closure{ .scalar = inputs[1].storage.scalar });
        // TODO(performance) fuse multiply and sum into single operation using map reduce
        const multiplied = try multiply(T, context.allocator, context.gradient_input, inputs[0]);
        outputs[1] = try sum(T, context.allocator, multiplied, null);
    }
    return outputs;
}

test "multiply backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 4);
    const y = try constant(f64, &arena.allocator, 10);
    const gradient_input = try constant(f64, &arena.allocator, 2);
    const actual = try multiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, 20);
    const expected_y_gradient = try constant(f64, &arena.allocator, 8);
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "multiply backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const y = try constant(f64, &arena.allocator, .{ 6, 7, 8, 9, 10 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try multiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, .{
        12, 28, 48, 72, 100,
    });
    const expected_y_gradient = try constant(f64, &arena.allocator, .{
        2, 8, 18, 32, 50,
    });
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "multiply backward rank 2" {
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
    const actual = try multiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected_x_gradient = try constant(f64, &arena.allocator, .{
        .{ 10, 24 },
        .{ 42, 64 },
    });
    const expected_y_gradient = try constant(f64, &arena.allocator, .{
        .{ 2, 8 },
        .{ 18, 32 },
    });
    expectEqual(f64, actual[0], expected_x_gradient);
    expectEqual(f64, actual[1], expected_y_gradient);
}

test "multiply backwards broadcast scalar rank 3" {
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
    const actual = try multiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ scalar, tensor },
    });
    const actual2 = try multiplyBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ tensor, scalar },
    });
    const expected_scalar_gradient = try constant(f64, &arena.allocator, -0.5);
    const expected_tensor_gradient = try constant(f64, &arena.allocator, .{
        .{
            .{ -0.625, -0.625 },
            .{ -0.625, -0.625 },
        },
        .{
            .{ -6.25e-01, -6.25e-01 },
            .{ -6.25e-01, -6.25e-01 },
        },
    });
    expectEqual(f64, actual[0], expected_scalar_gradient);
    expectEqual(f64, actual[1], expected_tensor_gradient);
    expectEqual(f64, actual2[0], expected_tensor_gradient);
    expectEqual(f64, actual2[1], expected_scalar_gradient);
}
