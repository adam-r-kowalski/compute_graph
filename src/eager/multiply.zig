const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

pub fn multiply(allocator: *Allocator, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    if (!std.mem.eql(usize, x.shape, y.shape))
        return error.ShapeMismatch;
    const T = @TypeOf(x);
    const shape = try allocator.alloc(usize, x.shape.len);
    errdefer allocator.free(shape);
    std.mem.copy(usize, shape, x.shape);
    const stride = try allocator.alloc(usize, x.stride.len);
    errdefer allocator.free(stride);
    std.mem.copy(usize, stride, x.stride);
    switch (x.storage) {
        .scalar => |scalar| {
            return T{
                .shape=shape,
                .stride=stride,
                .storage=.{.scalar = scalar * y.storage.scalar},
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T.ScalarType, array.len);
            errdefer allocator.free(new_array);
            const y_array = y.storage.array;
            for (array) |e, i| new_array[i] = e * y_array[i];
            return T{
                .shape=shape,
                .stride=stride,
                .storage=.{.array = new_array},
            };
        },
    }
}

test "multiply rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, 5));
    const y = try constant(&arena.allocator, @as(f64, 10));
    const z = try multiply(&arena.allocator, x, y);
    expect(std.mem.eql(usize, x.shape, z.shape));
    expect(std.mem.eql(usize, x.stride, z.stride));
    expectEqual(z.storage.scalar, 50);
}

test "multiply rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{1, -2, 3, -4, -5, 6});
    const y = try multiply(&arena.allocator, x, x);
    expect(std.mem.eql(usize, x.shape, y.shape));
    expect(std.mem.eql(usize, x.stride, y.stride));
    expect(std.mem.eql(i32, y.storage.array, &[_]i32{1, 4, 9, 16, 25, 36}));
}

test "multiply rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try multiply(&arena.allocator, x, x);
    expect(std.mem.eql(usize, x.shape, y.shape));
    expect(std.mem.eql(usize, x.stride, y.stride));
    expect(std.mem.eql(f16, y.storage.array, &[_]f16{1, 4, 9, 16, 25, 36}));
}

test "multiply rank 3" {
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
    const y = try multiply(&arena.allocator, x, x);
    expect(std.mem.eql(usize, x.shape, y.shape));
    expect(std.mem.eql(usize, x.stride, y.stride));
    expect(std.mem.eql(i8, y.storage.array, &[_]i8{1, 4, 9, 16, 25, 36, 49, 64}));
}
