const Node = @import("node.zig").Node;

pub fn Tensor(comptime S: type, comptime r: u64) type {
    return struct {
        node: Node,
    };
}
