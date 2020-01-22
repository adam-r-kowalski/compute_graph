const Allocator = @import("std").mem.Allocator;
const Node = @import("node.zig").Node;
const CpuTensorUnion = @import("../eager.zig").CpuTensorUnion;

pub const Operation = struct {
    pub const ForwardError = error{
        OutOfMemory,
        ShapeMismatch,
        Overflow,
    };

    pub const BackwardError = error{
        OutOfMemory,
        CannotDifferentiateIntegral,
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

    pub const ForwardResult = ForwardError!CpuTensorUnion;
    pub const BackwardResult = BackwardError![]const CpuTensorUnion;

    inputs: fn (self: *const Operation) []const Node,
    forward: fn (context: ForwardContext) ForwardResult,
    backward: ?fn (context: BackwardContext) BackwardResult,
};
