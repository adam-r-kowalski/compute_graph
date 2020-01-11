const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn CpuStorage(comptime ScalarType: type) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,


        fn deinit(self: @This(), allocator: *Allocator) void {
            switch (self) {
                .array => |array| allocator.free(array),
                else => {},
            }
        }
    };
}

pub fn CpuTensor(comptime ScalarType: type) type {
    return struct {
        shape: []const usize,
        stride: []const usize,
        storage: CpuStorage(ScalarType),

        pub fn deinit(self: @This(), allocator: *Allocator) void {
            allocator.free(self.shape);
            allocator.free(self.stride);
            self.storage.deinit(allocator);
        }
    };
}
