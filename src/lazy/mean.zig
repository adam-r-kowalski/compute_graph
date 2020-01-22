const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;

const Mean = struct {
    operation: Operation,
    nodes: [1]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Mean, "operation", operation).nodes;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{ .f64 = try eager.mean(context.allocator, tensor) },
        .f32 => |tensor| .{ .f32 = try eager.mean(context.allocator, tensor) },
        .f16 => |tensor| .{ .f16 = try eager.mean(context.allocator, tensor) },
        .i64 => |tensor| .{ .f64 = try eager.mean(context.allocator, tensor) },
        .i32 => |tensor| .{ .f32 = try eager.mean(context.allocator, tensor) },
        .i8 => |tensor| .{ .f16 = try eager.mean(context.allocator, tensor) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch(context.value) {
        .f64 => |tensor| { values[0] = .{ .f64 = tensor }; },
        .f32 => |tensor| { values[0] = .{ .f32 = tensor }; },
        .f16 => |tensor| { values[0] = .{ .f16 = tensor }; },
        .i64, .i32, .i8 => { return error.CannotDifferentiateIntegral; },
    }
    return values;
}

pub fn mean(graph: *Graph, x: var) !@TypeOf(x) {
    var mean_operation = try graph.arena.allocator.create(Mean);
    mean_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .nodes = .{x.node},
    };
    try graph.operations.append(&mean_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return @TypeOf(x){ .node = node };
}

test "mean scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try mean(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(y);
    const expected = try eager.constant(&arena.allocator, @as(f64, -5));
    expectEqual(actual.f64, expected);
}

test "mean matrix" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]f64{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const z = try mean(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, @as(f64, 8));
    expectEqual(actual.f64, expected);
}

test "mean matrix i32" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]i32{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const z = try mean(&graph, x);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(&arena.allocator, @as(f32, 8));
    expectEqual(actual.f32, expected);
}
