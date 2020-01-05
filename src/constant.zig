const std = @import("std");
const Node = @import("node.zig").Node;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const cpuTensor = @import("cpu_tensor.zig").cpuTensor;

fn TensorType(comptime T: type) type {
    return Tensor(f64, 0);
}

pub fn constant(graph: *Graph, value: var) !TensorType(@TypeOf(value)) {
    try graph.constants.append(.{ .value = value });
    const node = Node{ .constant = graph.constants.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "constant" {
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const x_out = try session.run(x);
    const y_out = try session.run(y);
    std.testing.expectEqual(x_out, 5);
    std.testing.expectEqual(y_out, 10);
}
