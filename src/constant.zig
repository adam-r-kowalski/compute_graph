const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const Node = @import("node.zig").Node;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;

pub fn constant(graph: *Graph, literal: var) !Tensor(f64, 0) {
    const tensor = try CpuTensor.init(&graph.arena.allocator, literal);
    try graph.constants.append(tensor);
    const node = Node{ .constant = graph.constants.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "constant scalar" {
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
    expectEqual(x_out.data.f64.scalar, 5);
    expectEqual(y_out.data.f64.scalar, 10);
}

test "constant array" {
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]f32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(x);
    const expected = try CpuTensor.init(allocator, [_][2]f32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    defer expected.deinit(allocator);
    expect(std.mem.eql(f32, actual.data.f32.array, expected.data.f32.array));
}
