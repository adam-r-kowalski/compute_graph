const Node = @import("node.zig").Node;

pub fn Tensor(comptime ScalarType: type, comptime rank: u64) type {
    return struct {
        node: Node,
    };
}
