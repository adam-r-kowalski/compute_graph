const std = @import("std");
const Node = @import("node.zig").Node;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const cpuTensor = @import("cpu_tensor.zig").cpuTensor;

pub fn constant(graph: *Graph, literal: var) !Tensor(f64, 0) {
    const tensor = try cpuTensor(&graph.arena.allocator, literal);
    try graph.constants.append(.{ .f64 = tensor });
    const node = Node{ .constant = graph.constants.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "constant" {
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 10));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const x_out = try session.run(x);
    const y_out = try session.run(y);
    std.testing.expectEqual(x_out, 5);
    std.testing.expectEqual(y_out, 10);
}
