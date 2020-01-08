const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const arrayInfo = @import("array_info.zig").arrayInfo;

pub fn TensorData(comptime ScalarType: type) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,

        fn transfer_memory(data: []ScalarType, literal: var, index: *usize) void {
            switch (@typeInfo(@TypeOf(literal))) {
                .Pointer, .Array => {
                    var i: usize = 0;
                    while (i < literal.len) : (i += 1)
                        transfer_memory(data, literal[i], index);
                },
                else => {
                    data[index.*] = literal;
                    index.* += @as(usize, 1);
                },
            }
        }

        fn init(comptime rank: usize, allocator: *Allocator, shape: []const usize, literal: var) error{OutOfMemory}!@This() {
            if (rank == 0)
                return @This(){ .scalar = literal };
            var data = try allocator.alloc(ScalarType, tensorLength(rank, shape));
            errdefer allocator.free(data);
            var index: usize = 0;
            transfer_memory(data, literal, &index);
            return @This(){ .array = data };
        }

        fn deinit(self: @This(), allocator: *Allocator) void {
            switch (self) {
                .array => |array| allocator.free(array),
                else => {},
            }
        }
    };
}

pub const CpuTensor = struct {
    pub const Data = union(enum) {
        f64: TensorData(f64),
        f32: TensorData(f32),
        f16: TensorData(f16),
        i64: TensorData(i64),
        i32: TensorData(i32),
        i8: TensorData(i8),
    };

    shape: []const usize,
    stride: []const usize,
    data: Data,

    pub fn deinit(self: CpuTensor, allocator: *Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.stride);
        switch (self.data) {
            .f64 => |tensor_data| tensor_data.deinit(allocator),
            .f32 => |tensor_data| tensor_data.deinit(allocator),
            .f16 => |tensor_data| tensor_data.deinit(allocator),
            .i64 => |tensor_data| tensor_data.deinit(allocator),
            .i32 => |tensor_data| tensor_data.deinit(allocator),
            .i8 => |tensor_data| tensor_data.deinit(allocator),
        }
    }

    pub fn init(allocator: *Allocator, literal: var) !CpuTensor {
        const T = arrayInfo(@TypeOf(literal));
        const shape = try tensorShape(T.rank, allocator, literal);
        const data = try TensorData(T.ScalarType).init(T.rank, allocator, shape, literal);
        return CpuTensor{
            .shape = shape,
            .stride = try tensorStride(T.rank, allocator, shape),
            .data = switch (T.ScalarType) {
                f64 => .{ .f64 = data },
                f32 => .{ .f32 = data },
                f16 => .{ .f16 = data },
                i64 => .{ .i64 = data },
                i32 => .{ .i32 = data },
                i8 => .{ .i8 = data },
                else => @compileError("CpuTensor ScalarType not supported"),
            },
        };
    }
};

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

fn tensorStride(comptime rank: usize, allocator: *Allocator, shape: []const usize) ![]usize {
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
    expect(std.mem.eql(usize, stride, &[_]usize{}));
}

test "stride rank 3" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 2, 3 };
    const stride = try tensorStride(3, allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{ 1, 3, 6 }));
}

test "stride rank 4" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 4, 5, 6 };
    const stride = try tensorStride(4, allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{ 1, 3, 12, 60 }));
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
    expectEqual(tensorLength(0, shape[0..]), 1);
}

test "length rank 3" {
    const shape = [_]usize{ 3, 2, 3 };
    expectEqual(tensorLength(3, shape[0..]), 18);
}

test "length rank 4" {
    const shape = [_]usize{ 3, 4, 5, 6 };
    expectEqual(tensorLength(4, shape[0..]), 360);
}

test "cpu tensor rank 0" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, @as(f16, 5));
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{}));
    expectEqual(tensor.data.f16.scalar, 5);
}

test "cpu tensor rank 1" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, &[_]f64{ 1, 2, 3 });
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{3}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{1}));
    expect(std.mem.eql(f64, tensor.data.f64.array, &[_]f64{ 1, 2, 3 }));
}

test "cpu tensor rank 2" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 2, 3 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 1, 2 }));
    expect(std.mem.eql(i32, tensor.data.i32.array, &[_]i32{ 1, 2, 3, 4, 5, 6 }));
}

test "cpu tensor rank 3" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, &[_][2][3]f16{
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
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 1, 3, 6 }));
    expect(std.mem.eql(f16, tensor.data.f16.array, &[_]f16{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    }));
}
