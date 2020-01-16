const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

pub fn CpuStorage(comptime ScalarType: type) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,
    };
}

pub fn CpuTensor(comptime T: type) type {
    return struct {
        shape: []const usize,
        stride: []const usize,
        storage: CpuStorage(T),

        pub const ScalarType = T;
    };
}

pub const CpuTensorUnion = union(enum) {
    f64: CpuTensor(f64),
    f32: CpuTensor(f32),
    f16: CpuTensor(f16),
    i64: CpuTensor(i64),
    i32: CpuTensor(i32),
    i8: CpuTensor(i8),

    pub fn init(tensor: var) CpuTensorUnion {
        return switch (@TypeOf(tensor).ScalarType) {
            f64 => .{ .f64 = tensor },
            f32 => .{ .f32 = tensor },
            f16 => .{ .f16 = tensor },
            i64 => .{ .i64 = tensor },
            i32 => .{ .i32 = tensor },
            i8 => .{ .i8 = tensor },
            else => @compileError("ScalarType not supported"),
        };
    }
};

pub fn tensorStride(comptime rank: usize, allocator: *Allocator, shape: []const usize) ![]usize {
    var stride = try allocator.alloc(usize, rank);
    errdefer allocator.free(stride);
    if (rank == 0)
        return stride;
    stride[rank - 1] = 1;
    var i: usize = rank - 1;
    var product: usize = 1;
    while (i > 0) : (i -= 1) {
        product *= shape[i];
        stride[i - 1] = product;
    }
    return stride;
}

test "stride rank 0" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{};
    const stride = try tensorStride(0, allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{}));
}

test "stride rank 3" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 2, 3 };
    const stride = try tensorStride(3, allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{ 6, 3, 1 }));
}

test "stride rank 4" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 4, 5, 6 };
    const stride = try tensorStride(4, allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{ 120, 30, 6, 1 }));
}

pub fn tensorLength(shape: []const usize) usize {
    var length: usize = 1;
    for (shape) |s| length *= s;
    return length;
}

test "length rank 0" {
    const shape = [_]usize{};
    expectEqual(tensorLength(shape[0..]), 1);
}

test "length rank 3" {
    const shape = [_]usize{ 3, 2, 3 };
    expectEqual(tensorLength(shape[0..]), 18);
}

test "length rank 4" {
    const shape = [_]usize{ 3, 4, 5, 6 };
    expectEqual(tensorLength(shape[0..]), 360);
}
