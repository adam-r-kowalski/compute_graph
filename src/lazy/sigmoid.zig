const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const sigmoidBackward = @import("../eager/sigmoid.zig").sigmoidBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

pub fn sigmoid(graph: *Graph, x: Tensor) !Tensor {
    const o = try onesLike(&graph, x);
    const n = try negate(&graph, x);
    const e = try exponentiate(&graph, n);
    const a = try add(&graph, o, e);
    return try divide(&graph, o, a);
}

test "sigmoid scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try sigmoid(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(f64, 5));
    expectEqual(f64, actual[0].f64, expected);
}

// test "sigmoid matrix" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var arena = std.heap.ArenaAllocator.init(allocator);
//     defer arena.deinit();
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]f64{
//         .{ 1, -2 },
//         .{ 3, -4 },
//         .{ -5, 6 },
//     });
//     const z = try sigmoid(&graph, x);
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(.{ .tensors = &[_]Tensor{z} });
//     const expected = try eager.constant(&arena.allocator, [_][2]f64{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     expectEqual(f64, actual[0].f64, expected);
// }

// test "sigmoid matrix i32" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var arena = std.heap.ArenaAllocator.init(allocator);
//     defer arena.deinit();
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]i32{
//         .{ 1, -2 },
//         .{ 3, -4 },
//         .{ -5, 6 },
//     });
//     const z = try sigmoid(&graph, x);
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(.{ .tensors = &[_]Tensor{z} });
//     const expected = try eager.constant(&arena.allocator, [_][2]i32{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     expectEqual(i32, actual[0].i32, expected);
// }

// test "gradient sigmoid" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const gradient = @import("gradient.zig").gradient;
//     const mean = @import("mean.zig").mean;
//     const allocator = std.heap.page_allocator;
//     var arena = std.heap.ArenaAllocator.init(allocator);
//     defer arena.deinit();
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const a = try constant(&graph, [_][2]f64{
//         .{ 0, -2 },
//         .{ 3, -4 },
//     });
//     const b = try sigmoid(&graph, a);
//     const c = try mean(&graph, b);
//     const gradients = try gradient(&graph, c, &[_]Tensor{a});
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(.{ .tensors = gradients });
//     const expected = try eager.constant(&arena.allocator, [_][2]f64{
//         .{ 0, -0.25 },
//         .{ 0.25, -0.25 },
//     });
//     expectEqual(f64, actual[0].f64, expected);
// }
