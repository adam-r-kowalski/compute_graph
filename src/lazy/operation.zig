const Allocator = @import("std").mem.Allocator;
const Node = @import("node.zig").Node;
const CpuTensorUnion = @import("../eager.zig").CpuTensorUnion;

pub const Operation = struct {
    pub const Error = error{
        OutOfMemory,
        ShapeMismatch,
        Overflow,
    };

    pub const Context = struct {
        op: *const Operation,
        allocator: *Allocator,
        values: []const CpuTensorUnion,
    };

    inputs: fn (self: *const Operation) []const Node,
    forward: fn (context: Context) Error!CpuTensorUnion,
};
