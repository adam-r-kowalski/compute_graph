const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

const Subtract = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    const self = @fieldParentPtr(Subtract, "operation", operation);
    return &self.nodes;
}

fn forward(operation: *const Operation, values: []const CpuTensor) CpuTensor {
    std.debug.assert(values.len == 2);
    // return values[0] - values[1];
    return values[0];
}

pub fn subtract(graph: *Graph, x: Tensor(f64, 0), y: Tensor(f64, 0)) !Tensor(f64, 0) {
    var subtract_operation = try graph.arena.allocator.create(Subtract);
    subtract_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
        },
        .nodes = .{ x.node, y.node },
    };
    try graph.operations.append(&subtract_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "subtract" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 10));
    const z = try subtract(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    // const z_out = try session.run(z);
    // std.testing.expectEqual(z_out, -5);
}
