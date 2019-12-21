const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;

const Multiply = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    const self = @fieldParentPtr(Multiply, "operation", operation);
    return &self.nodes;
}

fn forward(operation: *const Operation, values: []const f64) f64 {
    std.debug.assert(values.len == 2);
    return values[0] * values[1];
}

pub fn multiply(graph: *Graph, x: Tensor(f64, 0), y: Tensor(f64, 0)) !Tensor(f64, 0) {
    var multiply_operation = try graph.arena.allocator.create(Multiply);
    multiply_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
        },
        .nodes = .{ x.node, y.node },
    };
    try graph.operations.append(&multiply_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "multiply" {
    const constant = @import("constant.zig").constant;
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try multiply(&graph, x, y);
    const operation = graph.operations.at(z.node.operation);
    const nodes = operation.inputs(operation);
    const left = graph.constants.at(nodes[0].constant);
    const right = graph.constants.at(nodes[1].constant);
    std.testing.expectEqual(graph.constants.at(x.node.constant), left);
    std.testing.expectEqual(graph.constants.at(y.node.constant), right);
}
