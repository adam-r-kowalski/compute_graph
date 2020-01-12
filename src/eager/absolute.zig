const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;

fn absoluteScalar(x: var) error{Overflow}!@TypeOf(x) {
    return switch (@TypeOf(x)) {
        f64 => std.math.absFloat(x),
        f32 => std.math.absFloat(x),
        f16 => std.math.absFloat(x),
        i64 => try std.math.absInt(x),
        i32 => try std.math.absInt(x),
        i8 => try std.math.absInt(x),
        else => @compileError("ScalarType not supported"),
    };
}

pub fn absolute(allocator: *Allocator, tensor: var) !@TypeOf(tensor) {
    const T = @TypeOf(tensor);
    const shape = try allocator.alloc(usize, tensor.shape.len);
    errdefer allocator.free(shape);
    std.mem.copy(usize, shape, tensor.shape);
    const stride = try allocator.alloc(usize, tensor.stride.len);
    errdefer allocator.free(stride);
    std.mem.copy(usize, stride, tensor.stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return T{
                .shape=shape,
                .stride=stride,
                .storage=.{.scalar = try absoluteScalar(scalar)},
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T.ScalarType, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = try absoluteScalar(e);
            return T{
                .shape=shape,
                .stride=stride,
                .storage=.{.array = new_array},
            };
        },
    }
}

test "absolute rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try absolute(&arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, 5));
    expectEqual(actual, expected);
}

test "absolute rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{1, -2, 3, -4, -5, 6});
    const actual = try absolute(&arena.allocator, x);
    const expected = try constant(&arena.allocator, [_]i32{1, 2, 3, 4, 5, 6});
    expectEqual(actual, expected);
}

test "absolute rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try absolute(&arena.allocator, x);
    const expected = try constant(&arena.allocator,[_][2]f16{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(actual, expected);
}

test "absolute rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const actual = try absolute(&arena.allocator, x);
    const expected = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 1, 2 },
            .{ 3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    expectEqual(actual, expected);
}
