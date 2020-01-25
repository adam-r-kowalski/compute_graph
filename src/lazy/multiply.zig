const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;

const Multiply = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Multiply, "operation", operation).nodes;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.multiply(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.multiply(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.multiply(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.multiply(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.multiply(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.multiply(i8, context.allocator, tensor, y.i8) },
    };
}

pub fn multiply(graph: *Graph, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    var multiply_operation = try graph.arena.allocator.create(Multiply);
    multiply_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = null,
        },
        .nodes = .{ x.node, y.node },
    };
    try graph.operations.append(&multiply_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return @TypeOf(x){ .node = node };
}

test "multiply scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 10));
    const z = try multiply(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, @as(f64, 50));
    expectEqual(f64, actual.f64, expected);
}

test "multiply matrix" {
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
    const z = try multiply(&graph, x, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 4 },
        .{ 9, 16 },
        .{ 25, 36 },
    });
    expectEqual(f64, actual.f64, expected);
}

test "multiply matrix i32" {
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
    const z = try multiply(&graph, x, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]i32{
        .{ 1, 4 },
        .{ 9, 16 },
        .{ 25, 36 },
    });
    expectEqual(i32, actual.i32, expected);
}
