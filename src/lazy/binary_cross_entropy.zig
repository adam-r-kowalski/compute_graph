const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Session = @import("session.zig").Session;
const constant = @import("constant.zig").constant;
const logarithm = @import("logarithm.zig").logarithm;
const multiply = @import("multiply.zig").multiply;
const add = @import("add.zig").add;
const subtract = @import("subtract.zig").subtract;
const onesLike = @import("ones_like.zig").onesLike;
const sum = @import("sum.zig").sum;
const mean = @import("mean.zig").mean;
const negate = @import("negate.zig").negate;
const gradient = @import("gradient.zig").gradient;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;

pub fn binaryCrossEntropy(graph: *Graph, y: Tensor, y_hat: Tensor) !Tensor {
    const a = try onesLike(graph, y_hat);
    const b = try subtract(graph, a, y_hat);
    const c = try logarithm(graph, b, .{});
    const d = try subtract(graph, a, y);
    const e = try multiply(graph, c, d);
    const f = try logarithm(graph, y_hat, .{});
    const g = try multiply(graph, y, f);
    const h = try add(graph, g, e);
    const i = try mean(graph, h);
    return try negate(graph, i);
}

test "binary cross entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const y = try constant(f64, &graph, .{
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    });
    const y_hat = try constant(f64, &graph, .{
        0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3,
    });
    const loss = try binaryCrossEntropy(&graph, y, y_hat);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(loss);
    const expected = try eager.constant(f64, &arena.allocator, 0.247);
    expectEqual(f64, actual.f64, expected);
}

test "gradient cross entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const y = try constant(f64, &graph, .{
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    });
    const y_hat = try constant(f64, &graph, .{
        0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3,
    });
    const loss = try binaryCrossEntropy(&graph, y, y_hat);
    const gradients = try gradient(&graph, loss, &[_]Tensor{ y, y_hat });
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_y_gradient = try eager.constant(f64, &arena.allocator, .{
        -0.1386, -0.2197, -0.2197, -0.04054, -0.1386, 0.2197, 0.0405, 0.1386, 0.2197, 0.0847,
    });
    const expected_y_hat_gradient = try eager.constant(f64, &arena.allocator, .{
        -0.125, -0.1111, -0.1111, -0.1666, -0.125, 0.1111, 0.1666, 0.125, 0.1111, 0.1428,
    });
    expectEqual(f64, actual[0].f64, expected_y_gradient);
    expectEqual(f64, actual[1].f64, expected_y_hat_gradient);
}
