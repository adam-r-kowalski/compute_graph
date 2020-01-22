const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;

const Absolute = struct {
    operation: Operation,
    nodes: [1]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Absolute, "operation", operation).nodes;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{ .f64 = try eager.absolute(context.allocator, tensor) },
        .f32 => |tensor| .{ .f32 = try eager.absolute(context.allocator, tensor) },
        .f16 => |tensor| .{ .f16 = try eager.absolute(context.allocator, tensor) },
        .i64 => |tensor| .{ .i64 = try eager.absolute(context.allocator, tensor) },
        .i32 => |tensor| .{ .i32 = try eager.absolute(context.allocator, tensor) },
        .i8 => |tensor| .{ .i8 = try eager.absolute(context.allocator, tensor) },
    };
}

pub fn absolute(graph: *Graph, x: var) !@TypeOf(x) {
    var absolute_operation = try graph.arena.allocator.create(Absolute);
    absolute_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = null,
        },
        .nodes = .{x.node},
    };
    try graph.operations.append(&absolute_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return @TypeOf(x){ .node = node };
}

test "absolute scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try absolute(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(y);
    const expected = try eager.constant(&arena.allocator, @as(f64, 5));
    expectEqual(actual.f64, expected);
}

test "absolute matrix" {
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
    const z = try absolute(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(actual.f64, expected);
}

test "absolute matrix i32" {
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
    const z = try absolute(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]i32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(actual.i32, expected);
}
