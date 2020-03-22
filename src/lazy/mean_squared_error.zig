const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const expectEqual = @import("../testing.zig").expectEqual;
const subtract = @import("subtract.zig").subtract;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;
const power = @import("power.zig").power;
const eager = @import("../eager.zig");

pub fn meanSquaredError(graph: *Graph, y: Tensor, y_hat: Tensor) !Tensor {
    const a = try subtract(graph, y, y_hat);
    const b = try power(graph, a, 2);
    return try mean(graph, b);
}

test "meanSquaredError scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const y = try constant(f64, &graph, -5);
    const y_hat = try constant(f64, &graph, 10);
    const loss = try meanSquaredError(&graph, y, y_hat);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{loss});
    const expected = try eager.constant(f64, &arena.allocator, 225);
    expectEqual(f64, actual[0].f64, expected);
    std.testing.expect(std.mem.eql(usize, loss.shape, &[_]usize{}));
}

test "meanSquaredError matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const y = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y_hat = try constant(f64, &graph, .{
        .{ 2, 3 },
        .{ 3, 2 },
        .{ -5, 6 },
    });
    const loss = try meanSquaredError(&graph, y, y_hat);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{loss});
    const expected = try eager.constant(f64, &arena.allocator, 10.3333);
    expectEqual(f64, actual[0].f64, expected);
    std.testing.expect(std.mem.eql(usize, loss.shape, &[_]usize{}));
}

test "gradient meanSquaredError" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const y = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y_hat = try constant(f64, &graph, .{
        .{ 2, -2 },
        .{ 1, 4 },
    });
    const loss = try meanSquaredError(&graph, y, y_hat);
    const gradients = try gradient(&graph, loss, &[_]Tensor{ y, y_hat });
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_y = try eager.constant(f64, &arena.allocator, .{
        .{ -5.0e-01, 2.0e+00 },
        .{ 1.0e+00, 0.0e+00 },
    });
    const expected_y_hat = try eager.constant(f64, &arena.allocator, .{
        .{ 5.0e-01, -2.0e+00 },
        .{ -1.0e+00, -0.0e+00 },
    });
    expectEqual(f64, actual[0].f64, expected_y);
    expectEqual(f64, actual[1].f64, expected_y_hat);
    std.testing.expect(std.mem.eql(usize, loss.shape, &[_]usize{}));
}
