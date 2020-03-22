const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const exponentiate = @import("exponentiate.zig").exponentiate;
const sum = @import("sum.zig").sum;
const divide = @import("divide.zig").divide;
const expectEqual = @import("../testing.zig").expectEqual;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;
const maximum = @import("maximum.zig").maximum;
const subtract = @import("subtract.zig").subtract;
const eager = @import("../eager.zig");

const SoftmaxParameters = struct {
    dimension: ?usize = null,
};

fn softmax(graph: *Graph, x: Tensor, parameters: SoftmaxParameters) !Tensor {
    const a = try maximum(graph, x, .{
        .dimension = parameters.dimension,
        .keep_dimensions = true,
    });
    const b = try subtract(graph, x, a);
    const c = try exponentiate(graph, b);
    const d = try sum(graph, c, .{
        .dimension = parameters.dimension,
        .keep_dimensions = true,
    });
    return try divide(graph, c, d);
}

test "softmax vector" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{ 0.1, 0.2, 0.3, 0.4, 0.5 });
    const b = try softmax(&graph, a, .{});
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ .tensors = &[_]Tensor{ b, c, gradients[0] } });
    const expected = try eager.constant(f64, &arena.allocator, .{ 0.1621, 0.1792, 0.1980, 0.2188, 0.2419 });
    const expected1 = try eager.constant(f64, &arena.allocator, 0.2);
    const expected2 = try eager.constant(f64, &arena.allocator, .{
        -4.6512e-18, -5.1404e-18, -5.6810e-18, -6.2785e-18, 2.1751e-17,
    });
    expectEqual(f64, actual[0].f64, expected);
    expectEqual(f64, actual[1].f64, expected1);
    expectEqual(f64, actual[2].f64, expected2);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{5}));
}

test "softmax matrix dimension 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 0.1, 0.2, 0.3, 0.4, 0.5 },
        .{ -0.1, -0.2, -0.3, -0.4, -0.5 },
    });
    const b = try softmax(&graph, a, .{ .dimension = 0 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ .tensors = &[_]Tensor{ b, c, gradients[0] } });
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.5498, 0.5986, 0.6456, 0.6899, 0.7310 },
        .{ 0.4501, 0.4013, 0.3543, 0.3100, 0.2689 },
    });
    const expected1 = try eager.constant(f64, &arena.allocator, 0.4999);
    const expected2 = try eager.constant(f64, &arena.allocator, .{
        .{ 7.6686e-18, -1.6950e-17, 9.3665e-18, 0, 0 },
        .{ 6.2785e-18, -1.1362e-17, 5.1404e-18, 0, 0 },
    });
    expectEqual(f64, actual[0].f64, expected);
    expectEqual(f64, actual[1].f64, expected1);
    expectEqual(f64, actual[2].f64, expected2);
}

test "softmax matrix dimension 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 0.1, 0.2, 0.3, 0.4, 0.5 },
        .{ -0.1, -0.2, -0.3, -0.4, -0.5 },
    });
    const b = try softmax(&graph, a, .{ .dimension = 1 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ .tensors = &[_]Tensor{ b, c, gradients[0] } });
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.1621, 0.1791, 0.1980, 0.2188, 0.2418 },
        .{ 0.2418, 0.2188, 0.1980, 0.1791, 0.1621 },
    });
    const expected1 = try eager.constant(f64, &arena.allocator, 0.2);
    const expected2 = try eager.constant(f64, &arena.allocator, .{
        .{ -2.3256e-18, -2.5702e-18, -2.8405e-18, -3.1392e-18, 1.0875e-17 },
        .{ -3.1392e-18, -2.8405e-18, -2.5702e-18, -2.3256e-18, -2.1043e-18 },
    });
    expectEqual(f64, actual[0].f64, expected);
    expectEqual(f64, actual[1].f64, expected1);
    expectEqual(f64, actual[2].f64, expected2);
}
