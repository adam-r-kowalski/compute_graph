const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;

pub fn add(allocator: *Allocator, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    if (!std.mem.eql(usize, x.shape, y.shape))
        return error.ShapeMismatch;
    const T = @TypeOf(x);
    const shape = x.shape;
    const stride = x.stride;
    switch (x.storage) {
        .scalar => |scalar| {
            return T{
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = scalar + y.storage.scalar },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T.ScalarType, array.len);
            errdefer allocator.free(new_array);
            const y_array = y.storage.array;
            for (array) |e, i| new_array[i] = e + y_array[i];
            return T{
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "add rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, 5));
    const y = try constant(&arena.allocator, @as(f64, 10));
    const actual = try add(&arena.allocator, x, y);
    const expected = try constant(&arena.allocator, @as(f64, 15));
    expectEqual(actual, expected);
}

test "add rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{ 1, -2, 3, -4, -5, 6 });
    const actual = try add(&arena.allocator, x, x);
    const expected = try constant(&arena.allocator, [_]i32{ 2, -4, 6, -8, -10, 12 });
    expectEqual(actual, expected);
}

test "add rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try add(&arena.allocator, x, x);
    const expected = try constant(&arena.allocator, [_][2]f16{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(actual, expected);
}

test "add rank 3" {
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
    const actual = try add(&arena.allocator, x, x);
    const expected = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 2, -4 },
            .{ 6, -8 },
        },
        .{
            .{ 10, -12 },
            .{ 14, -16 },
        },
    });
    expectEqual(actual, expected);
}
