const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const exponentiate = @import("exponentiate.zig").exponentiate;
const sum = @import("sum.zig").sum;
const divide = @import("divide.zig").divide;
const expectEqual = @import("../testing.zig").expectEqual;

const SoftmaxParameters = struct {
    dimension: ?usize = null,
};

fn softmax(graph: *Graph, x: Tensor, parameters: SoftmaxParameters) !Tensor {
    const a = try exponentiate(graph, x);
    const b = try sum(graph, a, .{ .dimension = parameters.dimension });
    return try divide(graph, a, b);
}

test "softmax vector" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{ 0.1, 0.2, 0.3, 0.4, 0.5 });
    const b = try softmax(&graph, a, .{});
    const c = try sum(&graph, b, .{});
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{5}));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{ b, c }, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{ 0.1621, 0.1792, 0.1980, 0.2188, 0.2419 });
    const expected2 = try eager.constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0].f64, expected);
    expectEqual(f64, actual[1].f64, expected2);
}

test "softmax matrix dimension 0" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 0.1, 0.2, 0.3, 0.4, 0.5 },
        .{ -0.1, -0.2, -0.3, -0.4, -0.5 },
    });
    const b = try softmax(&graph, a, .{ .dimension = 0 });
    const c = try sum(&graph, b, .{ .dimension = 0 });
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 5 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{ b, c }, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.5498, 0.5987, 0.6457, 0.69, 0.7311 },
        .{ 0.4502, 0.4013, 0.3543, 0.31, 0.2689 },
    });
    const expected2 = try eager.constant(f64, &arena.allocator, .{ 1, 1, 1, 1, 1 });
    expectEqual(f64, actual[0].f64, expected);
    expectEqual(f64, actual[1].f64, expected2);
}

test "gradient softmax matrix dimension 0" {
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
    const a = try constant(f64, &graph, .{
        .{ 0.1, 0.2, 0.3, 0.4, 0.5 },
        .{ -0.1, -0.2, -0.3, -0.4, -0.5 },
    });
    const b = try softmax(&graph, a, .{ .dimension = 0 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 7.6686637462121e-18, -1.69503683056263e-17, 9.366527051024609e-18, 0, 0 },
        .{ 6.278570844035703e-18, -1.1362171662948268e-17, 5.140459035391771e-18, 0, 0 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
