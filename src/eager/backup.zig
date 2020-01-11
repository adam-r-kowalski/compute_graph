const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const assert = std.debug.assert;
const arrayInfo = @import("../util/array_info.zig").arrayInfo;

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

pub fn TypedCpuTensor(comptime ScalarType: type) type {
    return struct {
        shape: []const usize,
        stride: []const usize,
        data: TensorData(ScalarType),

        pub fn deinit(self: @This(), allocator: *Allocator) void {
            allocator.free(self.shape);
            allocator.free(self.stride);
            self.data.deinit(allocator);
        }

        pub fn init(comptime rank: usize, allocator: *Allocator, literal: var) !@This() {
            const shape = try tensorShape(rank, allocator, literal);
            return @This(){
                .shape = shape,
                .stride = try tensorStride(rank, allocator, shape),
                .data = try TensorData(ScalarType).init(rank, allocator, shape, literal),
            };
        }
    };
}

pub const CpuTensor = union(enum) {
    f64: TypedCpuTensor(f64),
    f32: TypedCpuTensor(f32),
    f16: TypedCpuTensor(f16),
    i64: TypedCpuTensor(i64),
    i32: TypedCpuTensor(i32),
    i8: TypedCpuTensor(i8),

    pub fn deinit(self: CpuTensor, allocator: *Allocator) void {
        switch (self) {
            .f64 => |tensor| tensor.deinit(allocator),
            .f32 => |tensor| tensor.deinit(allocator),
            .f16 => |tensor| tensor.deinit(allocator),
            .i64 => |tensor| tensor.deinit(allocator),
            .i32 => |tensor| tensor.deinit(allocator),
            .i8 => |tensor| tensor.deinit(allocator),
        }
    }

    pub fn init(allocator: *Allocator, literal: var) !CpuTensor {
        const T = arrayInfo(@TypeOf(literal));
        return switch (T.ScalarType) {
            f64 => .{ .f64 = try TypedCpuTensor(T.ScalarType).init(T.rank, allocator, literal) },
            f32 => .{ .f32 = try TypedCpuTensor(T.ScalarType).init(T.rank, allocator, literal) },
            f16 => .{ .f16 = try TypedCpuTensor(T.ScalarType).init(T.rank, allocator, literal) },
            i64 => .{ .i64 = try TypedCpuTensor(T.ScalarType).init(T.rank, allocator, literal) },
            i32 => .{ .i32 = try TypedCpuTensor(T.ScalarType).init(T.rank, allocator, literal) },
            i8 => .{ .i8 = try TypedCpuTensor(T.ScalarType).init(T.rank, allocator, literal) },
            else => @compileError("CpuTensor ScalarType not supported"),
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
    expect(std.mem.eql(usize, tensor.f16.shape, &[_]usize{}));
    expect(std.mem.eql(usize, tensor.f16.stride, &[_]usize{}));
    expectEqual(tensor.f16.data.scalar, 5);
}

test "cpu tensor rank 1" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, &[_]f64{ 1, 2, 3 });
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.f64.shape, &[_]usize{3}));
    expect(std.mem.eql(usize, tensor.f64.stride, &[_]usize{1}));
    expect(std.mem.eql(f64, tensor.f64.data.array, &[_]f64{ 1, 2, 3 }));
}

test "cpu tensor rank 2" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    defer tensor.deinit(allocator);
    expect(std.mem.eql(usize, tensor.i32.shape, &[_]usize{ 2, 3 }));
    expect(std.mem.eql(usize, tensor.i32.stride, &[_]usize{ 3, 1 }));
    expect(std.mem.eql(i32, tensor.i32.data.array, &[_]i32{ 1, 2, 3, 4, 5, 6 }));
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
    expect(std.mem.eql(usize, tensor.f16.shape, &[_]usize{ 3, 2, 3 }));
    expect(std.mem.eql(usize, tensor.f16.stride, &[_]usize{ 6, 3, 1 }));
    expect(std.mem.eql(f16, tensor.f16.data.array, &[_]f16{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    }));
}

pub fn get_index(comptime T: type, shape: []const usize, stride: []const usize, data: []const T, cartesian_index: []const usize) T {
    var linear_index: usize = 0;
    for (stride) |s, i| {
        assert(cartesian_index[i] < shape[i]);
        linear_index += s * cartesian_index[i];
    }
    return data[linear_index];
}

test "cpu tensor get_index" {
    const allocator = std.heap.page_allocator;
    const tensor = try CpuTensor.init(allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    defer tensor.deinit(allocator);
    const shape = tensor.i32.shape;
    const stride = tensor.i32.stride;
    const data = tensor.i32.data.array;
    expectEqual(get_index(i32, shape, stride, data, &[_]usize{ 0, 0 }), 1);
    expectEqual(get_index(i32, shape, stride, data, &[_]usize{ 0, 1 }), 2);
    expectEqual(get_index(i32, shape, stride, data, &[_]usize{ 0, 2 }), 3);
    expectEqual(get_index(i32, shape, stride, data, &[_]usize{ 1, 0 }), 4);
    expectEqual(get_index(i32, shape, stride, data, &[_]usize{ 1, 1 }), 5);
    expectEqual(get_index(i32, shape, stride, data, &[_]usize{ 1, 2 }), 6);
}
