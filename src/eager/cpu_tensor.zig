const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn CpuStorage(comptime ScalarType: type) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,
    };
}

pub fn CpuTensor(comptime T: type) type {
    return struct {
        shape: []const usize,
        stride: []const usize,
        storage: CpuStorage(T),

        pub const ScalarType = T;
    };
}

pub const CpuTensorUnion = union(enum) {
    f64: CpuTensor(f64),
    f32: CpuTensor(f32),
    f16: CpuTensor(f16),
    i64: CpuTensor(i64),
    i32: CpuTensor(i32),
    i8: CpuTensor(i8),

    pub fn init(tensor: var) CpuTensorUnion {
        return switch (@TypeOf(tensor).ScalarType) {
            f64 => .{.f64 = tensor},
            f32 => .{.f32 = tensor},
            f16 => .{.f16 = tensor},
            i64 => .{.i64 = tensor},
            i32 => .{.i32 = tensor},
            i8 => .{.i8 = tensor},
            else => @compileError("ScalarType not supported"),
        };
    }
};
