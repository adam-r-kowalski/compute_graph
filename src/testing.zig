const std = @import("std");
const cpu_tensor = @import("eager/cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;

pub fn expectEqual(comptime T: type, x: CpuTensor(T), y: CpuTensor(T)) void {
    std.testing.expect(std.mem.eql(usize, x.shape, y.shape));
    std.testing.expect(std.mem.eql(usize, x.stride, y.stride));
    switch (x.storage) {
        .scalar => |scalar| std.testing.expectEqual(scalar, y.storage.scalar),
        .array => |array| {
            const y_array = y.storage.array;
            for (array) |e, i| std.testing.expectEqual(e, y_array[i]);
        },
    }
}
