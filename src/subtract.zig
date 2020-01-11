const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Node = @import("node.zig").Node;
const Operation = @import("operation.zig").Operation;
const cpu_tensor = @import("cpu_tensor.zig");
const TensorData = cpu_tensor.TensorData;
const CpuTensor = cpu_tensor.CpuTensor;

const Subtract = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Subtract, "operation", operation).nodes;
}

fn forwardScalar(comptime T: type, x: T, y: T) CpuTensor.Data {
    const scalar = x - y;
    return switch (T) {
        f64 => .{ .f64 = .{ .scalar = scalar } },
        f32 => .{ .f32 = .{ .scalar = scalar } },
        f16 => .{ .f16 = .{ .scalar = scalar } },
        i64 => .{ .i64 = .{ .scalar = scalar } },
        i32 => .{ .i32 = .{ .scalar = scalar } },
        i8 => .{ .i8 = .{ .scalar = scalar } },
        else => @compileError("ScalarType not supported"),
    };
}

fn forwardArray(comptime T: type, allocator: *Allocator, x: []const T, y: []const T) !CpuTensor.Data {
    const array = try allocator.alloc(T, x.len);
    errdefer allocator.free(array);
    var i: usize = 0;
    while (i < x.len) : (i += 1)
        array[i] = x[i] - y[i];
    return switch (T) {
        f64 => CpuTensor.Data{ .f64 = .{ .array = array } },
        f32 => CpuTensor.Data{ .f32 = .{ .array = array } },
        f16 => CpuTensor.Data{ .f16 = .{ .array = array } },
        i64 => CpuTensor.Data{ .i64 = .{ .array = array } },
        i32 => CpuTensor.Data{ .i32 = .{ .array = array } },
        i8 => CpuTensor.Data{ .i8 = .{ .array = array } },
        else => @compileError("ScalarType not supported"),
    };
}

fn forwardData(comptime T: type, allocator: *Allocator, x: TensorData(T), y: TensorData(T)) !CpuTensor.Data {
    switch (x) {
        .scalar => |scalar| return forwardScalar(T, scalar, y.scalar),
        .array => |array| return try forwardArray(T, allocator, array, y.array),
    }
}

fn forward(context: Operation.Context) Operation.Error!CpuTensor {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    if (!std.mem.eql(usize, x.shape, y.shape))
        return error.ShapeMismatch;
    const shape = try context.allocator.alloc(usize, x.shape.len);
    errdefer context.allocator.free(shape);
    std.mem.copy(usize, shape, x.shape);
    const stride = try context.allocator.alloc(usize, x.stride.len);
    errdefer context.allocator.free(stride);
    std.mem.copy(usize, stride, x.stride);
    const data = blk: {
        switch (context.values[0].data) {
            .f64 => |data| break :blk try forwardData(f64, context.allocator, data, y.data.f64),
            .f32 => |data| break :blk try forwardData(f32, context.allocator, data, y.data.f32),
            .f16 => |data| break :blk try forwardData(f16, context.allocator, data, y.data.f16),
            .i64 => |data| break :blk try forwardData(i64, context.allocator, data, y.data.i64),
            .i32 => |data| break :blk try forwardData(i32, context.allocator, data, y.data.i32),
            .i8 => |data| break :blk try forwardData(i8, context.allocator, data, y.data.i8),
        }
    };
    return CpuTensor{ .shape = shape, .stride = stride, .data = data };
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

// test "subtract scalar" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, @as(f64, 5));
//     const y = try constant(&graph, @as(f64, 10));
//     const z = try subtract(&graph, x, y);
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const z_out = try session.run(z);
//     expectEqual(z_out.f64.data.scalar, -5);
// }

// test "subtract matrix" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]f64{
//         .{ 7, 8 },
//         .{ 9, 10 },
//         .{ 11, 12 },
//     });
//     expectEqual(@TypeOf(x), Tensor(f64, 2));
//     const y = try constant(&graph, [_][2]f64{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     expectEqual(@TypeOf(y), Tensor(f64, 2));
//     const z = try subtract(&graph, x, y);
//     expectEqual(@TypeOf(z), Tensor(f64, 2));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(z);
//     const expected = try CpuTensor.init(allocator, [_][2]f64{
//         .{ 6, 6 },
//         .{ 6, 6 },
//         .{ 6, 6 },
//     });
//     defer expected.deinit(allocator);
//     expect(std.mem.eql(f64, actual.f64.data.array, expected.f64.data.array));
// }

// test "subtract matrix i32" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]i32{
//         .{ 7, 8 },
//         .{ 9, 10 },
//         .{ 11, 12 },
//     });
//     expectEqual(@TypeOf(x), Tensor(i32, 2));
//     const y = try constant(&graph, [_][2]i32{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     expectEqual(@TypeOf(y), Tensor(i32, 2));
//     const z = try subtract(&graph, x, y);
//     expectEqual(@TypeOf(z), Tensor(i32, 2));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(z);
//     const expected = try CpuTensor.init(allocator, [_][2]i32{
//         .{ 6, 6 },
//         .{ 6, 6 },
//         .{ 6, 6 },
//     });
//     defer expected.deinit(allocator);
//     expect(std.mem.eql(i32, actual.i32.data.array, expected.i32.data.array));
// }
