const Node = @import("node.zig").Node;

pub const Operation = struct {
    inputs: fn (self: *const Operation) []const Node,
    forward: fn (self: *const Operation, values: []const f64) f64,
};
