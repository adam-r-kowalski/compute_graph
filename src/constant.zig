const std = @import("std");
const Node = @import("node.zig").Node;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;

pub fn constant(graph: *Graph, value: f64) !Tensor(f64, 0) {
    try graph.constants.append(.{ .value = value });
    const node = Node{ .constant = graph.constants.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "constant" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    std.testing.expectEqual(graph.constants.at(x.node.constant).value, 5);
    std.testing.expectEqual(graph.constants.at(y.node.constant).value, 10);
}
