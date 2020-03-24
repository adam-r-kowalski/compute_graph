const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const expectEqual = @import("../testing.zig").expectEqual;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;
const divide = @import("divide.zig").divide;
const eager = @import("../eager.zig");
const naturalLogarithm = @import("natural_logarithm.zig").naturalLogarithm;

pub fn logarithm(graph: *Graph, x: Tensor, parameters: var) !Tensor {
    if (@hasField(@TypeOf(parameters), "base") and parameters.@"base" != std.math.e) {
        const base = switch (x.scalarType) {
            .f64 => try constant(f64, graph, parameters.@"base"),
            .f32 => try constant(f32, graph, parameters.@"base"),
            .f16 => try constant(f16, graph, parameters.@"base"),
            .i64 => try constant(i64, graph, parameters.@"base"),
            .i32 => try constant(i32, graph, parameters.@"base"),
            .i8 => try constant(i8, graph, parameters.@"base"),
        };
        const a = try naturalLogarithm(graph, x);
        const b = try naturalLogarithm(graph, base);
        return try divide(graph, a, b);
    } else {
        return try naturalLogarithm(graph, x);
    }
}

test "logarithm scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, 5);
    const y = try logarithm(&graph, x, .{});
    const gradients = try gradient(&graph, y, &[_]Tensor{x});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ y, gradients[0] });
    const expected_y = try eager.constant(f64, &arena.allocator, 1.6094);
    const expected_gradients = try eager.constant(f64, &arena.allocator, 0.2);
    expectEqual(f64, actual[0].f64, expected_y);
    expectEqual(f64, actual[1].f64, expected_gradients);
    std.testing.expectEqual(y.shape, &[_]usize{});
}

test "logarithm matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const y = try logarithm(&graph, x, .{});
    const z = try mean(&graph, y);
    const gradients = try gradient(&graph, z, &[_]Tensor{x});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ y, gradients[0] });
    const expected_y = try eager.constant(f64, &arena.allocator, .{
        .{ 0, 0.6931 },
        .{ 1.0986, 1.3862 },
        .{ 1.6094, 1.7917 },
    });
    const expected_gradients = try eager.constant(f64, &arena.allocator, .{
        .{ 0.1666, 0.0833 },
        .{ 0.0555, 0.0416 },
        .{ 0.0333, 0.0277 },
    });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
    expectEqual(f64, actual[0].f64, expected_y);
    expectEqual(f64, actual[1].f64, expected_gradients);
}

test "logarithm base 2 scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, 5);
    const y = try logarithm(&graph, x, .{ .base = 2 });
    const gradients = try gradient(&graph, y, &[_]Tensor{x});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ y, gradients[0] });
    const expected_y = try eager.constant(f64, &arena.allocator, 2.3219);
    const expected_gradients = try eager.constant(f64, &arena.allocator, 0.2885);
    expectEqual(f64, actual[0].f64, expected_y);
    expectEqual(f64, actual[1].f64, expected_gradients);
    std.testing.expectEqual(y.shape, &[_]usize{});
}

test "logarithm base 2 matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const y = try logarithm(&graph, x, .{ .base = 2 });
    const z = try mean(&graph, y);
    const gradients = try gradient(&graph, z, &[_]Tensor{x});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ y, gradients[0] });
    const expected_y = try eager.constant(f64, &arena.allocator, .{
        .{ 0, 1 },
        .{ 1.5849, 2 },
        .{ 2.3219, 2.5849 },
    });
    const expected_gradients = try eager.constant(f64, &arena.allocator, .{
        .{ 0.2404, 0.1202 },
        .{ 0.0801, 0.0601 },
        .{ 0.0480, 0.0400 },
    });
    expectEqual(f64, actual[0].f64, expected_y);
    expectEqual(f64, actual[1].f64, expected_gradients);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
}
