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

const Absolute = struct {
    operation: Operation,
    nodes: [1]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Absolute, "operation", operation).nodes;
}

fn forward(context: Operation.Context) Operation.Error!CpuTensorUnion {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{.f64 = try eager.absolute(context.allocator, tensor)},
        .f32 => |tensor| .{.f32 = try eager.absolute(context.allocator, tensor)},
        .f16 => |tensor| .{.f16 = try eager.absolute(context.allocator, tensor)},
        .i64 => |tensor| .{.i64 = try eager.absolute(context.allocator, tensor)},
        .i32 => |tensor| .{.i32 = try eager.absolute(context.allocator, tensor)},
        .i8 => |tensor| .{.i8 = try eager.absolute(context.allocator, tensor)},
    };
}

pub fn absolute(graph: *Graph, x: var) !@TypeOf(x) {
    var absolute_operation = try graph.arena.allocator.create(Absolute);
    absolute_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
        },
        .nodes = .{ x.node },
    };
    try graph.operations.append(&absolute_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return @TypeOf(x){ .node = node };
}

test "absolute scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, @as(f64, 5));
    const b = try constant(&graph, @as(f64, -5));
    const c = try absolute(&graph, a);
    const d = try absolute(&graph, b);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const c_out = try session.run(c);
    expectEqual(c_out.f64.storage.scalar, 5);
    const d_out = try session.run(d);
    expectEqual(d_out.f64.storage.scalar, 5);
}

test "absolute matrix" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]f64{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    expectEqual(@TypeOf(x), Tensor(f64, 2));
    const z = try absolute(&graph, x);
    expectEqual(@TypeOf(z), Tensor(f64, 2));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    defer expected.deinit(allocator);
    expect(std.mem.eql(f64, actual.f64.storage.array, expected.storage.array));
}

test "absolute matrix i32" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, [_][2]i32{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    expectEqual(@TypeOf(x), Tensor(i32, 2));
    const z = try absolute(&graph, x);
    expectEqual(@TypeOf(z), Tensor(i32, 2));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(z);
    const expected = try eager.constant(allocator, [_][2]i32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    defer expected.deinit(allocator);
    expect(std.mem.eql(i32, actual.i32.storage.array, expected.storage.array));
}
