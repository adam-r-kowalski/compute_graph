const std = @import("std");
const Allocator = std.mem.Allocator;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const invoke = @import("invoke.zig").invoke;

pub fn map(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), invokable: var) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = invoke(invokable, .{scalar}) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = invoke(invokable, .{e});
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}
