const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const expectEqual = @import("../testing.zig").expectEqual;
const add = @import("add.zig").add;
const divide = @import("divide.zig").divide;
const exponentiate = @import("exponentiate.zig").exponentiate;
const onesLike = @import("ones_like.zig").onesLike;
const negate = @import("negate.zig").negate;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;
const eager = @import("../eager.zig");

pub fn sigmoid(graph: *Graph, x: Tensor) !Tensor {
    const a = try onesLike(graph, x);
    const b = try negate(graph, x);
    const c = try exponentiate(graph, b);
    const d = try add(graph, a, c);
    return try divide(graph, a, d);
}

test "sigmoid scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, -5);
    const y = try sigmoid(&graph, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(y);
    const expected = try eager.constant(f64, &arena.allocator, 0.0066);
    expectEqual(f64, actual.f64, expected);
    std.testing.expectEqual(y.shape, &[_]usize{});
}

test "sigmoid matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try sigmoid(&graph, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(y);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.7310, 0.1192 },
        .{ 0.9525, 0.0179 },
        .{ 0.0066, 0.9975 },
    });
    expectEqual(f64, actual.f64, expected);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
}

test "gradient sigmoid" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try sigmoid(&graph, a);
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.0492, 0.0262 },
        .{ 0.0113, 0.0044 },
    });
    expectEqual(f64, actual[0].f64, expected);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
}
