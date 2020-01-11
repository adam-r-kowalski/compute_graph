const std = @import("std");

const ArrayInfo = struct {
    ScalarType: type,
    rank: usize,
};

pub fn arrayInfo(comptime T: type) ArrayInfo {
    var ScalarType = T;
    var rank = 0;
    while (true) {
        switch (@typeInfo(ScalarType)) {
            .Pointer => |pointer| ScalarType = pointer.child,
            .Array => |array| {
                rank += 1;
                ScalarType = array.child;
            },
            else => return .{ .ScalarType = ScalarType, .rank = rank },
        }
    }
}

test "array info rank 0" {
    const info = arrayInfo(@TypeOf(2));
    std.testing.expectEqual(info, ArrayInfo{ .ScalarType = comptime_int, .rank = 0 });
}

test "array info rank 1" {
    const info = arrayInfo(@TypeOf(&[_]f64{ 1, 2, 3 }));
    std.testing.expectEqual(info, ArrayInfo{ .ScalarType = f64, .rank = 1 });
}

test "array info rank 2" {
    const info = arrayInfo(@TypeOf(&[_][3]i32{
        .{ 1, 2, 3 },
        .{ 1, 2, 3 },
    }));
    std.testing.expectEqual(info, ArrayInfo{ .ScalarType = i32, .rank = 2 });
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
    std.testing.expectEqual(info, ArrayInfo{ .ScalarType = f16, .rank = 3 });
}
