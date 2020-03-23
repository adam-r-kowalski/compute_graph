const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Session = @import("session.zig").Session;
const constant = @import("constant.zig").constant;
const logarithm = @import("logarithm.zig").logarithm;
const multiply = @import("multiply.zig").multiply;
const sum = @import("sum.zig").sum;
const divide = @import("divide.zig").divide;
const gradient = @import("gradient.zig").gradient;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;

pub fn klDivergence(graph: *Graph, p: Tensor, q: Tensor) !Tensor {
    const a = try divide(graph, p, q);
    const b = try logarithm(graph, a, .{ .base = 2 });
    const c = try multiply(graph, p, b);
    return try sum(graph, c, .{});
}

test "kl divergence" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const q = try constant(f64, &graph, .{ 0.8, 0.15, 0.05 });
    const kl = try klDivergence(&graph, p, q);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(kl);
    const expected = try eager.constant(f64, &arena.allocator, 1.9269);
    expectEqual(f64, actual.f64, expected);
}

test "kl divergence same probability distribution" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const kl = try klDivergence(&graph, p, p);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(kl);
    const expected = try eager.constant(f64, &arena.allocator, 0);
    expectEqual(f64, actual.f64, expected);
}

test "gradient kl divergence" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const q = try constant(f64, &graph, .{ 0.8, 0.15, 0.05 });
    const kl = try klDivergence(&graph, p, q);
    const gradients = try gradient(&graph, kl, &[_]Tensor{ p, q });
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_p_gradient = try eager.constant(f64, &arena.allocator, .{
        -1.5573, 2.8577, 4.7646,
    });
    const expected_q_gradient = try eager.constant(f64, &arena.allocator, .{
        -0.1803, -3.8471, -14.4269,
    });
    expectEqual(f64, actual[0].f64, expected_p_gradient);
    expectEqual(f64, actual[1].f64, expected_q_gradient);
}
