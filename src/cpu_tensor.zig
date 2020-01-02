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

pub fn CpuTensor(comptime T: type, comptime r: u64) type {
    return struct {
        const ScalarType = T;
        const rank = r;

        shape: [rank]u64,
        stride: [rank]u64,
        data: []const ScalarType,
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
    stride[0] = 1;
    var i: u64 = 1;
    var product: u64 = 1;
    while (i < rank) : (i += 1) {
        product *= shape[i - 1];
        stride[i] = product;
    }
    return stride;
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

pub fn cpuTensor(literal: var) tensorType(@TypeOf(literal)) {
    const T = tensorType(@TypeOf(literal));
    const shape = tensorShape(T.rank, literal);
    return T{
        .shape = shape,
        .stride = tensorStride(T.rank, shape),
        .data = undefined,
    };
}

test "cpu tensor rank 1" {
    const tensor = cpuTensor(&[_]f64{ 1, 2, 3 });
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{3}));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{1}));
}

test "cpu tensor rank 2" {
    const tensor = cpuTensor(&[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{ 2, 3 }));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{ 1, 2 }));
}

test "cpu tensor rank 3" {
    const tensor = cpuTensor(&[_][2][3]f16{
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
    std.testing.expect(std.mem.eql(u64, tensor.shape[0..], &[_]u64{ 3, 2, 3 }));
    std.testing.expect(std.mem.eql(u64, tensor.stride[0..], &[_]u64{ 1, 3, 6 }));
}
