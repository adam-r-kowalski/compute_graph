const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensorUnion = eager.CpuTensorUnion;

const Subtract = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Subtract, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.subtract(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.subtract(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.subtract(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.subtract(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.subtract(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.subtract(i8, context.allocator, tensor, y.i8) },
    };
}

pub fn subtract(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    var subtract_operation = try graph.arena.allocator.create(Subtract);
    subtract_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = null,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&subtract_operation.operation);
    return Tensor{ .operation = graph.operations.len - 1 };
}

test "subtract scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 10));
    const z = try subtract(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, @as(f64, -5));
    expectEqual(f64, actual.f64, expected);
}

test "subtract matrix" {
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
    const y = try constant(&graph, [_][2]f64{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    const z = try subtract(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f64, actual.f64, expected);
}

test "subtract matrix i32" {
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
    const y = try constant(&graph, [_][2]i32{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    const z = try subtract(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]i32{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(i32, actual.i32, expected);
}
