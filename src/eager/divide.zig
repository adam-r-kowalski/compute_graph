const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");
const onesLike = @import("ones_like.zig").onesLike;
const multiply = @import("multiply.zig").multiply;
const negate = @import("negate.zig").negate;
const zip = @import("broadcast.zig").zip;

fn TensorType(comptime T: type) type {
    const ScalarType = switch (T) {
        f64, f32, f16 => T,
        i64 => f64,
        i32 => f32,
        i8 => f16,
        else => @compileError("ScalarType not supported"),
    };
    return CpuTensor(ScalarType);
}

fn coerceToFloat(comptime T: type, x: var) T {
    return switch (@TypeOf(x)) {
        f64, f32, f16 => @as(T, x),
        else => @intToFloat(T, x),
    };
}

// TODO(alternative design choice) should I support integer division rather than convert to float?
pub fn divide(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !TensorType(T) {
    if (!std.mem.eql(usize, x.shape, y.shape))
        return error.ShapeMismatch;
    const Tensor = TensorType(T);
    const shape = x.shape;
    const stride = x.stride;
    switch (x.storage) {
        .scalar => |scalar| {
            return Tensor{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = coerceToFloat(Tensor.ScalarType, scalar) / coerceToFloat(Tensor.ScalarType, y.storage.scalar) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(Tensor.ScalarType, array.len);
            errdefer allocator.free(new_array);
            const y_array = y.storage.array;
            for (array) |e, i| new_array[i] = coerceToFloat(Tensor.ScalarType, e) / coerceToFloat(Tensor.ScalarType, y_array[i]);
            return Tensor{
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
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
    const expected = try constant(f32, &arena.allocator, .{ 0.1666, 0.4, 0.75, 1.3333, 2.5, 6 });
    expectEqual(f32, actual, expected);
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
    const expected = try constant(f16, &arena.allocator, .{
        .{
            .{ 0.125, 0.2856 },
            .{ 0.5, 0.7998 },
        },
        .{
            .{ 1.25, 2 },
            .{ 3.5, 8 },
        },
    });
    expectEqual(f16, actual, expected);
}

pub fn divideBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const outputs = try context.allocator.alloc(CpuTensor(T), 2);
    errdefer context.allocator.free(outputs);
    const a = context.forward_inputs[0];
    const b = context.forward_inputs[1];
    const g = context.gradient_input;
    const allocator = context.allocator;
    outputs[0] = blk: {
        const c = try onesLike(T, allocator, b);
        const d = try divide(T, allocator, c, b);
        break :blk try multiply(T, allocator, d, g);
    };
    outputs[1] = blk: {
        const c = try divide(T, allocator, a, b);
        const d = try divide(T, allocator, c, b);
        const e = try negate(T, allocator, d);
        break :blk try multiply(T, allocator, e, g);
    };
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
