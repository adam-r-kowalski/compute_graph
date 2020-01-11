const std = @import("std");
const Allocator = std.mem.Allocator;
const TypedCpuTensor = @import("backup.zig").TypedCpuTensor;

fn absoluteScalar(comptime T: type, x: T) error{Overflow}!T {
    return switch (T) {
        f64 => std.math.absFloat(x),
        f32 => std.math.absFloat(x),
        f16 => std.math.absFloat(x),
        i64 => try std.math.absInt(x),
        i32 => try std.math.absInt(x),
        i8 => try std.math.absInt(x),
        else => @compileError("ScalarType not supported"),
    };
}

pub fn absolute(comptime T: type, allocator: *Allocator, tensor: TypedCpuTensor(T)) !TypedCpuTensor(T) {
    const shape = try allocator.alloc(usize, tensor.shape.len);
    errdefer allocator.free(shape);
    std.mem.copy(usize, shape, tensor.shape);
    const stride = try allocator.alloc(usize, tensor.stride.len);
    errdefer allocator.free(stride);
    std.mem.copy(usize, stride, tensor.stride);
    switch (tensor.data) {
        .scalar => |scalar| {
            return TypedCpuTensor(T){
                .shape=shape,
                .stride=stride,
                .data=.{.scalar = try absoluteScalar(T, scalar)},
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = try absoluteScalar(T, e);
            return TypedCpuTensor(T){
                .shape=shape,
                .stride=stride,
                .data=.{.array = new_array},
            };
        },
    }
}

test "absolute matrix" {
    const allocator = std.heap.page_allocator;
    const x = try TypedCpuTensor(f64).init(2, allocator, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    defer x.deinit(allocator);
    const y = try absolute(f64, allocator, x);
    defer y.deinit(allocator);
}
