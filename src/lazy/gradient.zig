const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Node = @import("node.zig").Node;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;

pub const Gradient = struct {
    of: Node,
    with_respect_to: Node,
};

pub fn gradient(graph: *Graph, of: var, with_respect_to: var) !@TypeOf(with_respect_to) {
    try graph.gradients.append(.{
        .of = of.node,
        .with_respect_to = with_respect_to.node
    });
    const node = Node{ .gradient = graph.gradients.len - 1 };
    return @TypeOf(with_respect_to){ .node = node };
}

test "gradient mean" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, [_][3]f64{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const b = try mean(&graph, a);
    const c = try gradient(&graph, b, a);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(c);
    const expected = try eager.constant(&arena.allocator, [_][3]f64{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    // const expected = try eager.constant(&arena.allocator, [_][3]f64{
    //     .{ 1/6, 1/6, 1/6 },
    //     .{ 1/6, 1/6, 1/6 },
    // });
    expectEqual(actual.f64, expected);
}
