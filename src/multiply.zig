const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

const Multiply = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    const self = @fieldParentPtr(Multiply, "operation", operation);
    return &self.nodes;
}

fn forward(context: Operation.Context) Operation.Error!CpuTensor {
    std.debug.assert(context.values.len == 2);
    if (!std.mem.eql(usize, context.values[0].shape, context.values[1].shape))
        return error.ShapeMismatch;
    // return values[0] * values[1];
    return context.values[0];
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
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 10));
    const z = try multiply(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    // const z_out = try session.run(z);
    // std.testing.expectEqual(z_out, 50);
}
