const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const Node = @import("node.zig").Node;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const arrayInfo = @import("../util/array_info.zig").arrayInfo;

fn TensorType(comptime T: type) type {
    const info = arrayInfo(T);
    return Tensor(info.ScalarType, info.rank);
}

pub fn constant(graph: *Graph, literal: var) !TensorType(@TypeOf(literal)) {
    const tensor = try eager.constant(&graph.arena.allocator, literal);
    try graph.constants.append(CpuTensorUnion.init(tensor));
    const node = Node{ .constant = graph.constants.len - 1 };
    return TensorType(@TypeOf(literal)){ .node = node };
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
    expectEqual(x_out.f64.storage.scalar, 5);
    expectEqual(y_out.f64.storage.scalar, 10);
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
    const expected = try eager.constant(allocator, [_][2]f32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    defer expected.deinit(allocator);
    expect(std.mem.eql(f32, actual.f32.storage.array, expected.storage.array));
}
