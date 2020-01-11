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
const tensorStride = cpu_tensor.tensorStride;

const MatrixMultiply = struct {
    operation: Operation,
    nodes: [2]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(MatrixMultiply, "operation", operation).nodes;
}

fn forwardScalar(comptime T: type, x: T, y: T) CpuTensor.Data {
    const scalar = x * y;
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
        array[i] = x[i] * y[i];
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
    return switch (x) {
        .scalar => |scalar| forwardScalar(T, scalar, y.scalar),
        .array => |array| try forwardArray(T, allocator, array, y.array),
    };
}

fn forward(context: Operation.Context) Operation.Error!CpuTensor {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    if (x.shape[1] != y.shape[0])
        return error.ShapeMismatch;
    const shape = try context.allocator.alloc(usize, 2);
    errdefer context.allocator.free(shape);
    shape[0] = x.shape[0];
    shape[1] = y.shape[1];
    const stride = try tensorStride(2, context.allocator, shape);
    errdefer context.allocator.free(stride);
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

pub fn matrixMultiply(graph: *Graph, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    var multiply_operation = try graph.arena.allocator.create(MatrixMultiply);
    multiply_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
        },
        .nodes = .{ x.node, y.node },
    };
    try graph.operations.append(&multiply_operation.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return @TypeOf(x){ .node = node };
}

// test "matrix multiply" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]f64{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     expectEqual(@TypeOf(x), Tensor(f64, 2));
//     const y = try constant(&graph, [_][2]f64{
//         .{ 7, 8 },
//         .{ 9, 10 },
//         .{ 11, 12 },
//     });
//     expectEqual(@TypeOf(y), Tensor(f64, 2));
//     const z = try matrixMultiply(&graph, x, y);
//     expectEqual(@TypeOf(z), Tensor(f64, 2));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(z);
//     const expected = try CpuTensor.init(allocator, [_][2]f64{
//         .{ 7, 16 },
//         .{ 27, 40 },
//         .{ 55, 72 },
//     });
//     defer expected.deinit(allocator);
//     expect(std.mem.eql(f64, actual.data.f64.array, expected.data.f64.array));
// }

// test "matrix multiply i32" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]i32{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     expectEqual(@TypeOf(x), Tensor(i32, 2));
//     const y = try constant(&graph, [_][2]i32{
//         .{ 7, 8 },
//         .{ 9, 10 },
//         .{ 11, 12 },
//     });
//     expectEqual(@TypeOf(y), Tensor(i32, 2));
//     const z = try matrixMultiply(&graph, x, y);
//     expectEqual(@TypeOf(z), Tensor(i32, 2));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(z);
//     const expected = try CpuTensor.init(allocator, [_][2]i32{
//         .{ 7, 16 },
//         .{ 27, 40 },
//         .{ 55, 72 },
//     });
//     defer expected.deinit(allocator);
//     expect(std.mem.eql(i32, actual.data.i32.array, expected.data.i32.array));
// }
