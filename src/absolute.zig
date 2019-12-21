const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;

const Absolute = struct {
    operation: Operation,
    nodes: [1]Node,
};

fn inputs(operation: *const Operation) []const Node {
    const self = @fieldParentPtr(Absolute, "operation", operation);
    return &self.nodes;
}

fn forward(operation: *const Operation, values: []const f64) f64 {
    std.debug.assert(values.len == 1);
    return std.math.absFloat(values[0]);
}

pub fn absolute(graph: *Graph, x: Tensor(f64, 0)) !Tensor(f64, 0) {
    var absolute_operation = try graph.arena.allocator.create(Absolute);
    absolute_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
        },
        .nodes = .{x.node},
    };
    try graph.operations.append(&absolute_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "absolute" {
    const constant = @import("constant.zig").constant;
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try absolute(&graph, x);
    const operation = graph.operations.at(y.node.operation);
    const nodes = operation.inputs(operation);
    const value = graph.constants.at(nodes[0].constant);
    std.testing.expectEqual(graph.constants.at(x.node.constant), value);
}
