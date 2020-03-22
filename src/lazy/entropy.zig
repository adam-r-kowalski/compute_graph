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

pub fn entropy(graph: *Graph, p: Tensor) !Tensor {
    const a = try logarithm(graph, p, .{ .base = 2 });
    const b = try multiply(graph, p, a);
    const c = try sum(graph, b, .{});
    return try negate(graph, c);
}

test "entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const e = try entropy(&graph, p);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ .tensors = &[_]Tensor{e} });
    const expected = try eager.constant(f64, &arena.allocator, 1.361);
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient cross entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const p = try constant(f64, &graph, .{ 0.1, 0.4, 0.5 });
    const e = try entropy(&graph, p);
    const gradients = try gradient(&graph, e, &[_]Tensor{p});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ .tensors = gradients });
    const expected_p_gradient = try eager.constant(f64, &arena.allocator, .{ 1.8792, -0.1207, -0.4426 });
    expectEqual(f64, actual[0].f64, expected_p_gradient);
}
