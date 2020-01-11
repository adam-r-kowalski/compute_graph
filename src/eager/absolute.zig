const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

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
            return @TypeOf(tensor){
                .shape=shape,
                .stride=stride,
                .storage=.{.array = new_array},
            };
        },
    }
}

test "absolute rank 0" {
    const allocator = std.heap.page_allocator;
    const a = try constant(allocator, @as(f64, -5));
    defer a.deinit(allocator);
    const b = try constant(allocator, @as(f64, 5));
    defer b.deinit(allocator);
    const c = try absolute(allocator, a);
    defer c.deinit(allocator);
    const d = try absolute(allocator, b);
    defer d.deinit(allocator);
    expect(std.mem.eql(usize, a.shape, c.shape));
    expect(std.mem.eql(usize, b.shape, d.shape));
    expect(std.mem.eql(usize, a.stride, c.stride));
    expect(std.mem.eql(usize, b.stride, d.stride));
    expectEqual(c.storage.scalar, 5);
    expectEqual(d.storage.scalar, 5);
}

test "absolute rank 1" {
    const allocator = std.heap.page_allocator;
    const x = try constant(allocator, [_]i32{1, -2, 3, -4, -5, 6});
    defer x.deinit(allocator);
    const y = try absolute(allocator, x);
    defer y.deinit(allocator);
    expect(std.mem.eql(usize, x.shape, y.shape));
    expect(std.mem.eql(usize, x.stride, y.stride));
    expect(std.mem.eql(i32, y.storage.array, &[_]i32{1, 2, 3, 4, 5, 6}));
}

test "absolute rank 2" {
    const allocator = std.heap.page_allocator;
    const x = try constant(allocator, [_][2]f16{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    defer x.deinit(allocator);
    const y = try absolute(allocator, x);
    defer y.deinit(allocator);
    expect(std.mem.eql(usize, x.shape, y.shape));
    expect(std.mem.eql(usize, x.stride, y.stride));
    expect(std.mem.eql(f16, y.storage.array, &[_]f16{1, 2, 3, 4, 5, 6}));
}

test "absolute rank 3" {
    const allocator = std.heap.page_allocator;
    const x = try constant(allocator, [_][2][2]i8{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    defer x.deinit(allocator);
    const y = try absolute(allocator, x);
    defer y.deinit(allocator);
    expect(std.mem.eql(usize, x.shape, y.shape));
    expect(std.mem.eql(usize, x.stride, y.stride));
    expect(std.mem.eql(i8, y.storage.array, &[_]i8{1, 2, 3, 4, 5, 6, 7, 8}));
}
