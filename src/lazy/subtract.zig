const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;

const Subtract = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Subtract, "operation", operation).nodes;
}

fn forward(context: Operation.Context) Operation.Error!CpuTensorUnion {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{.f64 = try eager.subtract(context.allocator, tensor, y.f64)},
        .f32 => |tensor| .{.f32 = try eager.subtract(context.allocator, tensor, y.f32)},
        .f16 => |tensor| .{.f16 = try eager.subtract(context.allocator, tensor, y.f16)},
        .i64 => |tensor| .{.i64 = try eager.subtract(context.allocator, tensor, y.i64)},
        .i32 => |tensor| .{.i32 = try eager.subtract(context.allocator, tensor, y.i32)},
        .i8 => |tensor| .{.i8 = try eager.subtract(context.allocator, tensor, y.i8)},
    };
}

pub fn subtract(graph: *Graph, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    var subtract_operation = try graph.arena.allocator.create(Subtract);
    subtract_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
        },
        .nodes = .{ x.node, y.node },
    };
    try graph.operations.append(&subtract_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return @TypeOf(x){ .node = node };
}

test "subtract scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 10));
    const z = try subtract(&graph, x, y);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const z_out = try session.run(z);
    expectEqual(z_out.f64.storage.scalar, -5);
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
    expectEqual(@TypeOf(x), Tensor(f64, 2));
    const y = try constant(&graph, [_][2]f64{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    expectEqual(@TypeOf(y), Tensor(f64, 2));
    const z = try subtract(&graph, x, y);
    expectEqual(@TypeOf(z), Tensor(f64, 2));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expect(std.mem.eql(f64, actual.f64.storage.array, expected.storage.array));
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
    expectEqual(@TypeOf(x), Tensor(i32, 2));
    const y = try constant(&graph, [_][2]i32{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    expectEqual(@TypeOf(y), Tensor(i32, 2));
    const z = try subtract(&graph, x, y);
    expectEqual(@TypeOf(z), Tensor(i32, 2));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, [_][2]i32{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expect(std.mem.eql(i32, actual.i32.storage.array, expected.storage.array));
}
