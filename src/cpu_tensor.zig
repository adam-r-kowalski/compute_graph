const std = @import("std");
const arrayInfo = @import("array_info.zig").arrayInfo;

fn TensorData(comptime ScalarType: type) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,

        fn init(comptime rank: usize, allocator: *std.mem.Allocator, shape: []const usize, literal: var) error{OutOfMemory}!@This() {
            if (rank == 0)
                return @This(){ .scalar = literal };

            var data = try allocator.alloc(ScalarType, tensorLength(rank, shape));
            errdefer allocator.free(data);
            var index: usize = 0;

            const Closure = struct {
                fn call(d: []ScalarType, l: var, i: *usize) void {
                    switch (@typeInfo(@TypeOf(l))) {
                        .Pointer, .Array => {
                            var j: usize = 0;
                            while (j < l.len) : (j += 1)
                                call(d, l[j], i);
                        },
                        else => {
                            d[i.*] = l;
                            i.* += @as(usize, 1);
                        },
                    }
                }
            };

            Closure.call(data, literal, &index);
            return @This(){ .array = data };
        }
    };
}

pub fn CpuTensor(comptime ScalarType: type) type {
    return struct {
        shape: []const usize,
        stride: []const usize,
        data: TensorData(ScalarType),

        pub fn deinit(self: @This(), allocator: *std.mem.Allocator) void {
            allocator.free(self.shape);
            allocator.free(self.stride);
            switch (self.data) {
                .array => |array| allocator.free(array),
                else => {},
            }
        }
    };
}

fn tensorType(comptime T: type) type {
    return CpuTensor(arrayInfo(T).ScalarType);
}

fn tensorShape(comptime rank: usize, allocator: *std.mem.Allocator, literal: var) ![]usize {
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

fn tensorStride(comptime rank: usize, allocator: *std.mem.Allocator, shape: []const usize) ![]usize {
    var stride = try allocator.alloc(usize, rank);
    errdefer allocator.free(stride);
    if (rank == 0)
        return stride;
    stride[0] = 1;
    var i: usize = 1;
    var product: usize = 1;
    while (i < rank) : (i += 1) {
        product *= shape[i - 1];
        stride[i] = product;
    }
    return stride;
}

test "stride rank 0" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{};
    const stride = try tensorStride(0, allocator, shape[0..]);
    defer allocator.free(stride);
    std.testing.expect(std.mem.eql(usize, stride, &[_]usize{}));
}

test "stride rank 3" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 2, 3 };
    const stride = try tensorStride(3, allocator, shape[0..]);
    defer allocator.free(stride);
    std.testing.expect(std.mem.eql(usize, stride, &[_]usize{ 1, 3, 6 }));
}

test "stride rank 4" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 4, 5, 6 };
    const stride = try tensorStride(4, allocator, shape[0..]);
    defer allocator.free(stride);
    std.testing.expect(std.mem.eql(usize, stride, &[_]usize{ 1, 3, 12, 60 }));
}

fn tensorLength(comptime rank: usize, shape: []const usize) usize {
    var i: usize = 0;
    var product: usize = 1;
    while (i < rank) : (i += 1)
        product *= shape[i];
    return product;
}

test "length rank 0" {
    const shape = [_]usize{};
    std.testing.expectEqual(tensorLength(0, shape[0..]), 1);
}

test "length rank 3" {
    const shape = [_]usize{ 3, 2, 3 };
    std.testing.expectEqual(tensorLength(3, shape[0..]), 18);
}

test "length rank 4" {
    const shape = [_]usize{ 3, 4, 5, 6 };
    std.testing.expectEqual(tensorLength(4, shape[0..]), 360);
}

pub fn cpuTensor(allocator: *std.mem.Allocator, literal: var) !tensorType(@TypeOf(literal)) {
    const T = arrayInfo(@TypeOf(literal));
    const shape = try tensorShape(T.rank, allocator, literal);
    return CpuTensor(T.ScalarType){
        .shape = shape,
        .stride = try tensorStride(T.rank, allocator, shape),
        .data = try TensorData(T.ScalarType).init(T.rank, allocator, shape, literal),
    };
}

test "cpu tensor rank 0" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, @as(f16, 5));
    defer tensor.deinit(allocator);
    std.testing.expect(std.mem.eql(usize, tensor.shape, &[_]usize{}));
    std.testing.expect(std.mem.eql(usize, tensor.stride, &[_]usize{}));
    std.testing.expectEqual(tensor.data.scalar, 5);
}

test "cpu tensor rank 1" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, &[_]f64{ 1, 2, 3 });
    defer tensor.deinit(allocator);
    std.testing.expect(std.mem.eql(usize, tensor.shape, &[_]usize{3}));
    std.testing.expect(std.mem.eql(usize, tensor.stride, &[_]usize{1}));
    std.testing.expect(std.mem.eql(f64, tensor.data.array, &[_]f64{ 1, 2, 3 }));
}

test "cpu tensor rank 2" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    defer tensor.deinit(allocator);
    std.testing.expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 2, 3 }));
    std.testing.expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 1, 2 }));
    std.testing.expect(std.mem.eql(i32, tensor.data.array, &[_]i32{ 1, 2, 3, 4, 5, 6 }));
}

test "cpu tensor rank 3" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, &[_][2][3]f16{
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
    std.testing.expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 3, 2, 3 }));
    std.testing.expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 1, 3, 6 }));
    std.testing.expect(std.mem.eql(f16, tensor.data.array, &[_]f16{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    }));
}
