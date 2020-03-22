const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Session = @import("session.zig").Session;
const constant = @import("constant.zig").constant;
const logarithm = @import("logarithm.zig").logarithm;
const multiply = @import("multiply.zig").multiply;
const sum = @import("sum.zig").sum;
const negate = @import("negate.zig").negate;
const gradient = @import("gradient.zig").gradient;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;

pub fn crossEntropy(graph: *Graph, p: Tensor, q: Tensor) !Tensor {
    const a = try logarithm(graph, q, .{ .base = 2 });
    const b = try multiply(graph, p, a);
    const c = try sum(graph, b, .{});
    return try negate(graph, c);
}

test "cross entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const q = try constant(f64, &graph, .{ 0.8, 0.15, 0.05 });
    const h_pq = try crossEntropy(&graph, p, q);
    const h_qp = try crossEntropy(&graph, q, p);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{ h_pq, h_qp });
    const expected_h_pq = try eager.constant(f64, &arena.allocator, 3.2879);
    const expected_h_qp = try eager.constant(f64, &arena.allocator, 2.9058);
    expectEqual(f64, actual[0].f64, expected_h_pq);
    expectEqual(f64, actual[1].f64, expected_h_qp);
}

test "gradient cross entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const q = try constant(f64, &graph, .{ 0.8, 0.15, 0.05 });
    const h = try crossEntropy(&graph, p, q);
    const gradients = try gradient(&graph, h, &[_]Tensor{ p, q });
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_p_gradient = try eager.constant(f64, &arena.allocator, .{
        0.3219, 2.7369, 4.3219,
    });
    const expected_q_gradient = try eager.constant(f64, &arena.allocator, .{
        -0.1803, -3.8471, -14.4269,
    });
    expectEqual(f64, actual[0].f64, expected_p_gradient);
    expectEqual(f64, actual[1].f64, expected_q_gradient);
}

test "cross entropy same distribution" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const q = try constant(f64, &graph, .{ 0.8, 0.15, 0.05 });
    const h_pp = try crossEntropy(&graph, p, p);
    const h_qq = try crossEntropy(&graph, q, q);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{ h_pp, h_qq });
    const expected_h_pp = try eager.constant(f64, &arena.allocator, 1.3609);
    const expected_h_qq = try eager.constant(f64, &arena.allocator, 0.8841);
    expectEqual(f64, actual[0].f64, expected_h_pp);
    expectEqual(f64, actual[1].f64, expected_h_qq);
}

test "gradient cross entropy same distribution" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const h = try crossEntropy(&graph, p, p);
    const gradients = try gradient(&graph, h, &[_]Tensor{p});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_p_gradient = try eager.constant(f64, &arena.allocator, .{
        1.8792, -0.1207, -0.4426,
    });
    expectEqual(f64, actual[0].f64, expected_p_gradient);
}
