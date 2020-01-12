const std = @import("std");
const cpu_tensor = @std("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;

pub fn expectEqual(x: var, y: @TypeOf(x)) void {
    const T = @TypeOf(x).ScalarType;
    std.testing.expect(std.mem.eql(usize, x.shape, y.shape));
    std.testing.expect(std.mem.eql(usize, x.stride, y.stride));
    switch (x.storage) {
        .scalar => |scalar| std.testing.expectEqual(scalar, y.storage.scalar),
        .array => |array| std.testing.expect(std.mem.eql(T, array, y.storage.array)),
    }
}
