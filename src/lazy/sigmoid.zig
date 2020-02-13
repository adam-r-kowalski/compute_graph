const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const expectEqual = @import("../testing.zig").expectEqual;
const add = @import("add.zig").add;
const divide = @import("divide.zig").divide;
const exponentiate = @import("exponentiate.zig").exponentiate;
const onesLike = @import("ones_like.zig").onesLike;
const negate = @import("negate.zig").negate;

pub fn sigmoid(graph: *Graph, x: Tensor) !Tensor {
    const a = try onesLike(graph, x);
    const b = try negate(graph, x);
    const c = try exponentiate(graph, b);
    const d = try add(graph, a, c);
    return try divide(graph, a, d);
}

test "sigmoid scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try sigmoid(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(f64, 0.0066));
    expectEqual(f64, actual[0].f64, expected);
}

test "sigmoid matrix" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const z = try sigmoid(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{z} });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 0.7310, 0.1192 },
        .{ 0.9525, 0.0179 },
        .{ 0.0066, 0.9975 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sigmoid" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const mean = @import("mean.zig").mean;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try sigmoid(&graph, a);
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = gradients });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 0.0492, 0.0262 },
        .{ 0.0113, 0.0044 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
