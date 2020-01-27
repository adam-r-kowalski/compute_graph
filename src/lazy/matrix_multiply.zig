const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;

const MatrixMultiply = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(MatrixMultiply, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.matrix_multiply(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.matrix_multiply(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.matrix_multiply(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.matrix_multiply(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.matrix_multiply(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.matrix_multiply(i8, context.allocator, tensor, y.i8) },
    };
}

pub fn matrix_multiply(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    var matrix_multiply_operation = try graph.arena.allocator.create(MatrixMultiply);
    matrix_multiply_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = null,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&matrix_multiply_operation.operation);
    return Tensor{ .operation = graph.operations.len - 1 };
}

test "matrix_multiply identity" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    });
    const y = try constant(&graph, [_][1]f64{
        .{1},
        .{2},
        .{3},
    });
    const z = try matrix_multiply(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z});
    const expected = try eager.constant(&arena.allocator, [_][1]f64{
        .{1},
        .{2},
        .{3},
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "matrix_multiply flip" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0, -1, 0 },
        .{ 0, 0, 1 },
    });
    const y = try constant(&graph, [_][1]f64{
        .{1},
        .{2},
        .{3},
    });
    const z = try matrix_multiply(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z});
    const expected = try eager.constant(&arena.allocator, [_][1]f64{
        .{1},
        .{-2},
        .{3},
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "matrix_multiply flip" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][3]f64{
        .{ 1, 2, 3 },
        .{ 4, 2, 5 },
        .{ 9, 8, 4 },
        .{ 6, 5, 3 },
    });
    const y = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 4, 6 },
        .{ 3, 9 },
    });
    const z = try matrix_multiply(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z});
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 18, 41 },
        .{ 27, 65 },
        .{ 53, 102 },
        .{ 35, 69 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
