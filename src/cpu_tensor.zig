const std = @import("std");

const ArrayInfo = struct {
    rank: u64,
    child: type,
};

fn arrayInfo(comptime T: type) ArrayInfo {
    var rank = 0;
    var child = T;
    while (true) {
        switch (@typeInfo(child)) {
            .Pointer => |pointer| child = pointer.child,
            .Array => |array| {
                rank += 1;
                child = array.child;
            },
            else => return .{ .rank = rank, .child = child },
        }
    }
}

test "array info rank 0" {
    const info = arrayInfo(@TypeOf(2));
    std.testing.expectEqual(info, ArrayInfo{ .rank = 0, .child = comptime_int });
}

test "array info rank 1" {
    const info = arrayInfo(@TypeOf(&[_]f64{ 1, 2, 3 }));
    std.testing.expectEqual(info, ArrayInfo{ .rank = 1, .child = f64 });
}

test "array info rank 2" {
    const info = arrayInfo(@TypeOf(&[_][3]i32{
        .{ 1, 2, 3 },
        .{ 1, 2, 3 },
    }));
    std.testing.expectEqual(info, ArrayInfo{ .rank = 2, .child = i32 });
}

test "array info rank 3" {
    const info = arrayInfo(@TypeOf(&[_][2][3]f16{
        .{
            .{ 1, 2, 3 },
            .{ 1, 2, 3 },
        },
        .{
            .{ 1, 2, 3 },
            .{ 1, 2, 3 },
        },
    }));
    std.testing.expectEqual(info, ArrayInfo{ .rank = 3, .child = f16 });
}

fn TensorData(comptime ScalarType: type, comptime rank: usize) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,

        fn init(allocator: *std.mem.Allocator, shape: [rank]u64, literal: var) error{OutOfMemory}!@This() {
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

pub fn CpuTensor(comptime T: type, comptime r: u64) type {
    return struct {
        const ScalarType = T;
        const rank = r;

        shape: [rank]u64,
        stride: [rank]u64,
        data: TensorData(ScalarType, rank),

        pub fn deinit(self: @This(), allocator: *std.mem.Allocator) void {
            if (rank > 0)
                allocator.free(self.data.array);
        }
    };
}

fn tensorType(comptime T: type) type {
    const info = arrayInfo(T);
    return CpuTensor(info.child, info.rank);
}

fn tensorShape(comptime rank: u64, literal: var) [rank]u64 {
    var shape: [rank]u64 = undefined;
    const Closure = struct {
        fn call(s: *[rank]u64, i: u64, l: var) void {
            switch (@typeInfo(@TypeOf(l))) {
                .Pointer, .Array => {
                    s[i] = l.len;
                    call(s, i + 1, l[0]);
                },
                else => {},
            }
        }
    };
    Closure.call(&shape, 0, literal);
    return shape;
}

fn tensorStride(comptime rank: u64, shape: [rank]u64) [rank]u64 {
    var stride: [rank]u64 = undefined;
    if (rank == 0)
        return stride;
    stride[0] = 1;
    var i: u64 = 1;
    var product: u64 = 1;
    while (i < rank) : (i += 1) {
        product *= shape[i - 1];
        stride[i] = product;
    }
    return stride;
}

test "stride rank 0" {
    const shape = [_]u64{};
    const stride = tensorStride(0, shape);
    std.testing.expect(std.mem.eql(u64, &stride, &[_]u64{}));
}

test "stride rank 3" {
    const shape = [_]u64{ 3, 2, 3 };
    const stride = tensorStride(3, shape);
    std.testing.expect(std.mem.eql(u64, &stride, &[_]u64{ 1, 3, 6 }));
}

test "stride rank 4" {
    const shape = [_]u64{ 3, 4, 5, 6 };
    const stride = tensorStride(4, shape);
    std.testing.expect(std.mem.eql(u64, &stride, &[_]u64{ 1, 3, 12, 60 }));
}

fn tensorLength(comptime rank: u64, shape: [rank]u64) u64 {
    var i: u64 = 0;
    var product: u64 = 1;
    while (i < rank) : (i += 1)
        product *= shape[i];
    return product;
}

test "length rank 0" {
    const shape = [_]u64{};
    std.testing.expectEqual(tensorLength(0, shape), 1);
}

test "length rank 3" {
    const shape = [_]u64{ 3, 2, 3 };
    std.testing.expectEqual(tensorLength(3, shape), 18);
}

test "length rank 4" {
    const shape = [_]u64{ 3, 4, 5, 6 };
    std.testing.expectEqual(tensorLength(4, shape), 360);
}

pub fn cpuTensor(allocator: *std.mem.Allocator, literal: var) !tensorType(@TypeOf(literal)) {
    const T = arrayInfo(@TypeOf(literal));
    const shape = tensorShape(T.rank, literal);

    return CpuTensor(T.child, T.rank){
        .shape = shape,
        .stride = tensorStride(T.rank, shape),
        .data = try TensorData(T.child, T.rank).init(allocator, shape, literal),
    };
}

test "cpu tensor rank 0" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, @as(f16, 5));
    defer tensor.deinit(allocator);
    std.testing.expectEqual(@TypeOf(tensor).ScalarType, f16);
    std.testing.expectEqual(@TypeOf(tensor).rank, 0);
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{}));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{}));
    std.testing.expectEqual(tensor.data.scalar, 5);
}

test "cpu tensor rank 1" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, &[_]f64{ 1, 2, 3 });
    defer tensor.deinit(allocator);
    std.testing.expectEqual(@TypeOf(tensor).ScalarType, f64);
    std.testing.expectEqual(@TypeOf(tensor).rank, 1);
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{3}));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{1}));
    std.testing.expect(std.mem.eql(f64, tensor.data.array, &[_]f64{ 1, 2, 3 }));
}

test "cpu tensor rank 2" {
    const allocator = std.heap.page_allocator;
    const tensor = try cpuTensor(allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    defer tensor.deinit(allocator);
    std.testing.expectEqual(@TypeOf(tensor).ScalarType, i32);
    std.testing.expectEqual(@TypeOf(tensor).rank, 2);
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{ 2, 3 }));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{ 1, 2 }));
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
    std.testing.expectEqual(@TypeOf(tensor).ScalarType, f16);
    std.testing.expectEqual(@TypeOf(tensor).rank, 3);
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{ 3, 2, 3 }));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{ 1, 3, 6 }));
    std.testing.expect(std.mem.eql(f16, tensor.data.array, &[_]f16{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    }));
}
