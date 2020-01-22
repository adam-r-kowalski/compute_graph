const Allocator = @import("std").mem.Allocator;
const Node = @import("node.zig").Node;
const CpuTensorUnion = @import("../eager.zig").CpuTensorUnion;

pub const Operation = struct {
    pub const Error = error{
        OutOfMemory,
        ShapeMismatch,
        Overflow,
    };

    pub const ForwardContext = struct {
        op: *const Operation,
        allocator: *Allocator,
        values: []const CpuTensorUnion,
    };

    pub const BackwardContext = struct {
        op: *const Operation,
        allocator: *Allocator,
        value: CpuTensorUnion,
    };

    inputs: fn (self: *const Operation) []const Node,
    forward: fn (context: ForwardContext) Error!CpuTensorUnion,
    backward: ?fn (context: BackwardContext) Error![]const CpuTensorUnion,
};
