const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const onesLikeBackward = @import("../eager/onesLike.zig").onesLikeBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const OnesLike = struct {
    operation: Operation,
    inputs: [1]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(OnesLike, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{ .f64 = try eager.onesLike(f64, context.allocator, tensor) },
        .f32 => |tensor| .{ .f32 = try eager.onesLike(f32, context.allocator, tensor) },
        .f16 => |tensor| .{ .f16 = try eager.onesLike(f16, context.allocator, tensor) },
        .i64 => |tensor| .{ .i64 = try eager.onesLike(i64, context.allocator, tensor) },
        .i32 => |tensor| .{ .i32 = try eager.onesLike(i32, context.allocator, tensor) },
        .i8 => |tensor| .{ .i8 = try eager.onesLike(i8, context.allocator, tensor) },
    };
}

pub fn onesLike(graph: *Graph, x: Tensor) !Tensor {
    var onesLike_operation = try graph.arena.allocator.create(OnesLike);
    onesLike_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = null,
        },
        .inputs = .{x},
    };
    try graph.operations.append(&onesLike_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = x.shape,
        .scalarType = x.scalarType,
    };
}

test "onesLike scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try onesLike(&graph, x);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(f64, 1));
    expectEqual(f64, actual[0].f64, expected);
}

test "onesLike matrix" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try onesLike(&graph, x);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 1 },
        .{ 1, 1 },
        .{ 1, 1 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "onesLike matrix i32" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]i32{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try onesLike(&graph, x);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, [_][2]i32{
        .{ 1, 1 },
        .{ 1, 1 },
        .{ 1, 1 },
    });
    expectEqual(i32, actual[0].i32, expected);
}
