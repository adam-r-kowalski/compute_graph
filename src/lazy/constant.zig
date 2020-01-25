const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const arrayInfo = @import("../util/array_info.zig").arrayInfo;
const expectEqual = @import("../testing.zig").expectEqual;

pub fn constant(graph: *Graph, literal: var) !Tensor {
    const tensor = try eager.constant(&graph.arena.allocator, literal);
    try graph.constants.append(CpuTensorUnion.init(tensor));
    return Tensor{ .constant = graph.constants.len - 1 };
}

test "constant scalar" {
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(x);
    const expected = try eager.constant(&arena.allocator, @as(f64, 5));
    expectEqual(f64, actual.f64, expected);
}

test "constant array" {
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
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
    const expected = try eager.constant(&arena.allocator, [_][2]f32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f32, actual.f32, expected);
}
