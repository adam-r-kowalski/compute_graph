const Node = @import("node.zig").Node;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

pub const Operation = struct {
    inputs: fn (self: *const Operation) []const Node,
    forward: fn (self: *const Operation, values: []const CpuTensor) CpuTensor,
};
