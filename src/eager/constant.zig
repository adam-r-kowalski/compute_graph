const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const arrayInfo = @import("../util/array_info.zig").arrayInfo;
const cpu_tensor = @import("cpu_tensor.zig"); 
const CpuTensor = cpu_tensor.CpuTensor;
const CpuStorage = cpu_tensor.CpuStorage;

fn ConstantType(comptime T: type) type {
    return CpuTensor(arrayInfo(T).ScalarType);
}

fn transferMemory(comptime ScalarType: type, array: []ScalarType, literal: var, index: *usize) void {
    switch (@typeInfo(@TypeOf(literal))) {
        .Pointer, .Array => {
            var i: usize = 0;
            while (i < literal.len) : (i += 1)
                transferMemory(ScalarType, array, literal[i], index);
        },
        else => {
            array[index.*] = literal;
            index.* += @as(usize, 1);
        },
    }
}

fn tensorShape(comptime rank: usize, allocator: *Allocator, literal: var) ![]usize {
    var shape = try allocator.alloc(usize, rank);
    errdefer allocator.free(shape);
    const Closure = struct {
        fn call(s: []usize, i: usize, l: var) void {
            switch (@typeInfo(@TypeOf(l))) {
                .Pointer, .Array => {
                    s[i] = l.len;
                    call(s, i + 1, l[0]);
                },
                else => {},
            }
        }
    };
    Closure.call(shape, 0, literal);
    return shape;
}

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

fn tensorLength(shape: []const usize) usize {
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

pub fn constant(allocator: *Allocator, literal: var) !ConstantType(@TypeOf(literal)) {
    const info = arrayInfo(@TypeOf(literal));
    const T = CpuTensor(info.ScalarType);
    const shape = try tensorShape(info.rank, allocator, literal);
    const stride = try tensorStride(info.rank, allocator, shape);
    if (info.rank == 0) {
        return T{
            .shape = shape,
            .stride = stride,
            .storage = CpuStorage(info.ScalarType){.scalar = literal},
        };
    }
    var array = try allocator.alloc(info.ScalarType, tensorLength(shape));
    errdefer allocator.free(array);
    var index: usize = 0;
    transferMemory(info.ScalarType, array, literal, &index);
    return T{
        .shape = shape,
        .stride = stride,
        .storage = CpuStorage(info.ScalarType){.array = array},
    };
}

test "constant rank 0" {
    const allocator = std.heap.page_allocator;
    const tensor = try constant(allocator, @as(f16, 5));
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{}));
    expectEqual(tensor.storage.scalar, 5);
}

test "constant rank 1" {
    const allocator = std.heap.page_allocator;
    const tensor = try constant(allocator, &[_]f64{ 1, 2, 3 });
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{3}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{1}));
    expect(std.mem.eql(f64, tensor.storage.array, &[_]f64{ 1, 2, 3 }));
}

test "constant rank 2" {
    const allocator = std.heap.page_allocator;
    const tensor = try constant(allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 2, 3 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 3, 1 }));
    expect(std.mem.eql(i32, tensor.storage.array, &[_]i32{ 1, 2, 3, 4, 5, 6 }));
}

test "constant rank 3" {
    const allocator = std.heap.page_allocator;
    const tensor = try constant(allocator, &[_][2][3]f16{
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
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 3, 2, 3 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 6, 3, 1 }));
    expect(std.mem.eql(f16, tensor.storage.array, &[_]f16{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    }));
}


