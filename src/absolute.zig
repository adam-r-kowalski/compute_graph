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
const TypedCpuTensor = cpu_tensor.TypedCpuTensor;
const CpuTensor = cpu_tensor.CpuTensor;

const Absolute = struct {
    operation: Operation,
    nodes: [1]Node,
};

fn inputs(operation: *const Operation) []const Node {
    return &@fieldParentPtr(Absolute, "operation", operation).nodes;
}

fn abs(comptime T: type, x: T) error{Overflow}!T {
    return switch (T) {
        f64 => std.math.absFloat(x),
        f32 => std.math.absFloat(x),
        f16 => std.math.absFloat(x),
        i64 => try std.math.absInt(x),
        i32 => try std.math.absInt(x),
        i8 => try std.math.absInt(x),
        else => @compileError("ScalarType not supported"),
    };
}

fn forwardScalar(comptime T: type, x: T) !TypedCpuTensor.Data {
    const scalar = try abs(T, x);
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

fn forwardArray(comptime T: type, allocator: *Allocator, x: []const T) !TypedCpuTensor.Data {
    const array = try allocator.alloc(T, x.len);
    errdefer allocator.free(array);
    var i: usize = 0;
    while (i < x.len) : (i += 1)
        array[i] = try abs(T, x[i]);
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

fn forwardData(comptime T: type, allocator: *Allocator, x: TensorData(T)) !CpuTensor.Data {
    switch (x) {
        .scalar => |scalar| return try forwardScalar(T, scalar),
        .array => |array| return try forwardArray(T, allocator, array),
    }
}

fn forward(context: Operation.Context) Operation.Error!CpuTensor {
    std.debug.assert(context.values.len == 1);
    const x = context.values[0];
    const shape = try context.allocator.alloc(usize, x.shape.len);
    errdefer context.allocator.free(shape);
    std.mem.copy(usize, shape, x.shape);
    const stride = try context.allocator.alloc(usize, x.stride.len);
    errdefer context.allocator.free(stride);
    std.mem.copy(usize, stride, x.stride);
    const data = blk: {
        switch (context.values[0].data) {
            .f64 => |data| break :blk try forwardData(f64, context.allocator, data),
            .f32 => |data| break :blk try forwardData(f32, context.allocator, data),
            .f16 => |data| break :blk try forwardData(f16, context.allocator, data),
            .i64 => |data| break :blk try forwardData(i64, context.allocator, data),
            .i32 => |data| break :blk try forwardData(i32, context.allocator, data),
            .i8 => |data| break :blk try forwardData(i8, context.allocator, data),
        }
    };
    return CpuTensor{ .shape = shape, .stride = stride, .data = data };
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

// test "absolute scalar" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const a = try constant(&graph, @as(f64, 5));
//     const b = try constant(&graph, @as(f64, -5));
//     const c = try absolute(&graph, a);
//     const d = try absolute(&graph, b);
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const c_out = try session.run(c);
//     expectEqual(c_out.f64.data.scalar, 5);
//     const d_out = try session.run(d);
//     expectEqual(d_out.f64.data.scalar, 5);
// }

// test "absolute matrix" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]f64{
//         .{ 1, -2 },
//         .{ 3, -4 },
//         .{ -5, 6 },
//     });
//     expectEqual(@TypeOf(x), Tensor(f64, 2));
//     const z = try absolute(&graph, x);
//     expectEqual(@TypeOf(z), Tensor(f64, 2));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(z);
//     const expected = try CpuTensor.init(allocator, [_][2]f64{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     defer expected.deinit(allocator);
//     expect(std.mem.eql(f64, actual.f64.data.array, expected.f64.data.array));
// }

// test "absolute matrix i32" {
//     const constant = @import("constant.zig").constant;
//     const Session = @import("session.zig").Session;
//     const allocator = std.heap.page_allocator;
//     var graph = try Graph.init(allocator);
//     defer graph.deinit();
//     const x = try constant(&graph, [_][2]i32{
//         .{ 1, -2 },
//         .{ 3, -4 },
//         .{ -5, 6 },
//     });
//     expectEqual(@TypeOf(x), Tensor(i32, 2));
//     const z = try absolute(&graph, x);
//     expectEqual(@TypeOf(z), Tensor(i32, 2));
//     var session = try Session.init(allocator, &graph);
//     defer session.deinit();
//     const actual = try session.run(z);
//     const expected = try CpuTensor.init(allocator, [_][2]i32{
//         .{ 1, 2 },
//         .{ 3, 4 },
//         .{ 5, 6 },
//     });
//     defer expected.deinit(allocator);
//     expect(std.mem.eql(i32, actual.i32.data.array, expected.i32.data.array));
// }
