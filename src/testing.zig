const std = @import("std");
const cpu_tensor = @import("eager/cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;

fn compareScalar(comptime T: type, actual: T, expected: T) void {
    switch (T) {
        i64, i32, i8 => std.testing.expectEqual(actual, expected),
        f64, f32, f16 => {
            if (!std.math.approxEq(T, actual, expected, 0.0001))
                std.debug.panic("expected {} found {}", .{ expected, actual });
        },
        else => @compileError("ScalarType not supported"),
    }
}

pub fn expectEqual(comptime T: type, actual: CpuTensor(T), expected: CpuTensor(T)) void {
    std.testing.expect(std.mem.eql(usize, actual.shape, expected.shape));
    std.testing.expect(std.mem.eql(usize, actual.stride, expected.stride));
    switch (expected.storage) {
        .scalar => |scalar| compareScalar(T, scalar, actual.storage.scalar),
        .array => |array| {
            const expected_array = expected.storage.array;
            for (array) |e, i| compareScalar(T, e, expected_array[i]);
        },
    }
}
