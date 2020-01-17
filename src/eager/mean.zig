const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;

fn TensorType(comptime T: type) type {
    const ScalarType = switch (T.ScalarType) {
        f64, f32, f16 => T.ScalarType,
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

pub fn mean(allocator: *Allocator, tensor: var) !TensorType(@TypeOf(tensor)) {
    const T = TensorType(@TypeOf(tensor));
    const shape = try allocator.alloc(usize, 0);
    errdefer allocator.free(shape);
    const stride = try allocator.alloc(usize, 0);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return T{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = coerceToFloat(T.ScalarType, scalar) },
            };
        },
        .array => |array| {
            var sum: T.ScalarType = 0;
            for (array) |e| sum += coerceToFloat(T.ScalarType, e);
            return T{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = sum / coerceToFloat(T.ScalarType, array.len) },
            };
        },
    }
}

test "mean rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, -5));
    expectEqual(actual, expected);
}

test "mean rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{ 5, 10, 7, 8, 10});
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f32, 8));
    expectEqual(actual, expected);
}

test "mean rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f16, 8));
    expectEqual(actual, expected);
}

test "mean rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 5, 10 },
            .{ 7, 8 },
        },
        .{
            .{ 10, 8 },
            .{ 2, 6 },
        },
    });
    const actual = try mean(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f16, 7));
    expectEqual(actual, expected);
}
