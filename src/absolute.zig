const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

const Absolute = struct {
    operation: Operation,
    nodes: [1]Node,
};

fn inputs(operation: *const Operation) []const Node {
    const self = @fieldParentPtr(Absolute, "operation", operation);
    return &self.nodes;
}

fn forward(context: Operation.Context) Operation.Error!CpuTensor {
    std.debug.assert(context.values.len == 1);
    // return std.math.absFloat(values[0]);
    return context.values[0];
}

pub fn absolute(graph: *Graph, x: Tensor(f64, 0)) !Tensor(f64, 0) {
    var absolute_operation = try graph.arena.allocator.create(Absolute);
    errdefer graph.arena.allocator.destroy(absolute_operation);
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
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, @as(f64, 5));
    const b = try constant(&graph, @as(f64, -5));
    const c = try absolute(&graph, a);
    const d = try absolute(&graph, b);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    // const c_out = try session.run(c);
    // const d_out = try session.run(d);
    // std.testing.expectEqual(c_out, 5);
    // std.testing.expectEqual(d_out, 5);
}
