const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const expectEqual = @import("../testing.zig").expectEqual;
const exponentiate = @import("exponentiate.zig").exponentiate;
const sum = @import("sum.zig").sum;
const divide = @import("divide.zig").divide;

pub fn softmax(graph: *Graph, x: Tensor, dimension: usize) !Tensor {
    const a = try exponentiate(graph, x);
    const b = try sum(graph, a, dimension);
    return try divide(graph, a, b);
}

// test "softmax vector" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const eager = @import("../eager.zig");
//     const allocator = std.heap.page_allocator;
//     var arena = std.heap.ArenaAllocator.init(allocator);
//     defer arena.deinit();
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_]f64{ -5, -3, 1, 2 });
//     const y = try softmax(&graph, x, 0);
//     std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 4 }));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
//     std.debug.warn("\n{}\n", .{actual[0]});
//     const expected = try eager.constant(&arena.allocator, [_][2]f64{
//         .{ 0.7310, 0.1192 },
//         .{ 0.9525, 0.0179 },
//         .{ 0.0066, 0.9975 },
//     });
//     expectEqual(f64, actual[0].f64, expected);
// }

// test "gradient softmax" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const gradient = @import("gradient.zig").gradient;
//     const mean = @import("mean.zig").mean;
//     const eager = @import("../eager.zig");
//     const allocator = std.heap.page_allocator;
//     var arena = std.heap.ArenaAllocator.init(allocator);
//     defer arena.deinit();
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const a = try constant(&graph, [_][2]f64{
//         .{ 1, 2 },
//         .{ 3, 4 },
//     });
//     const b = try softmax(&graph, a);
//     std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
//     const c = try mean(&graph, b);
//     const gradients = try gradient(&graph, c, &[_]Tensor{a});
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(.{ .tensors = gradients });
//     const expected = try eager.constant(&arena.allocator, [_][2]f64{
//         .{ 0.0492, 0.0262 },
//         .{ 0.0113, 0.0044 },
//     });
//     expectEqual(f64, actual[0].f64, expected);
// }
