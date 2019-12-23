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

pub fn CpuTensor(comptime ScalarType: type, comptime rank: u64) type {
    return struct {
        stride: [rank]u64,
        data: []ScalarType,
    };
}

fn tensorType(comptime T: type) type {
    const info = arrayInfo(T);
    return CpuTensor(info.child, info.rank);
}

pub fn cpuTensor(literal: var) tensorType(@TypeOf(literal)) {
    const T = tensorType(@TypeOf(literal));
    return T{ .stride = undefined, .data = literal };
}

test "cpu tensor" {
    const tensor = cpuTensor(&[_]f64{ 1, 2, 3 });
    std.testing.expect(std.mem.eql(f64, tensor.data, &[_]f64{ 1, 2, 3 }));
}
