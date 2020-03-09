const Allocator = @import("std").mem.Allocator;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

pub fn Context(comptime T: type) type {
    return struct {
        allocator: *Allocator,
        gradient_input: CpuTensor(T),
        forward_inputs: []const CpuTensor(T),
        forward_output: CpuTensor(T),
    };
}
