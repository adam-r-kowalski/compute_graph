const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Node = @import("node.zig").Node;

pub const Gradient = struct {
    of: Node,
    with_respect_to: Node,
};

pub fn gradient(graph: *Graph, of: var, with_respect_to: var) !@TypeOf(with_respect_to) {
    try graph.gradients.append(.{
        .of = of.node,
        .with_respect_to = with_respect_to.node
    });
    const node = Node{ .gradient = graph.gradients.len - 1 };
    return @TypeOf(with_respect_to){ .node = node };
}
