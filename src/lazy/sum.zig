const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const ScalarType = tensor.ScalarType;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const sumBackward = @import("../eager/sum.zig").sumBackward;
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const Sum = struct {
    operation: Operation,
    inputs: [1]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Sum, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |t| .{ .f64 = try eager.sum(f64, context.allocator, t, null) },
        .f32 => |t| .{ .f32 = try eager.sum(f32, context.allocator, t, null) },
        .f16 => |t| .{ .f16 = try eager.sum(f16, context.allocator, t, null) },
        .i64 => |t| .{ .i64 = try eager.sum(i64, context.allocator, t, null) },
        .i32 => |t| .{ .i32 = try eager.sum(i32, context.allocator, t, null) },
        .i8 => |t| .{ .i8 = try eager.sum(i8, context.allocator, t, null) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try sumBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try sumBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => |gradient_input| {
            const gradients = try sumBackward(f16, EagerBackwardContext(f16){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f16){context.forward_inputs[0].f16},
            });
            values[0] = .{ .f16 = gradients[0] };
        },
        .i64, .i32, .i8 => {
            return error.CannotDifferentiateIntegral;
        },
    }
    return values;
}

pub fn sum(graph: *Graph, x: Tensor, dimension: ?usize) !Tensor {
    var sum_operation = try graph.arena.allocator.create(Sum);
    sum_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
    };
    try graph.operations.append(&sum_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = &[_]usize{},
        .scalarType = x.scalarType,
    };
}

test "sum scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try sum(&graph, x, null);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(f64, -5));
    expectEqual(f64, actual[0].f64, expected);
}

test "sum matrix" {
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
    const y = try sum(&graph, x, null);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(f64, 48));
    expectEqual(f64, actual[0].f64, expected);
}

test "sum matrix i32" {
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
    const y = try sum(&graph, x, null);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(i32, 48));
    expectEqual(i32, actual[0].i32, expected);
}

test "gradient sum" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try sum(&graph, a, null);
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{gradients[0]} });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 1 },
        .{ 1, 1 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sum with multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const multiply = @import("multiply.zig").multiply;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try sum(&graph, a, null);
    const c = try constant(&graph, @as(f64, 5));
    const d = try multiply(&graph, b, c);
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, d, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{gradients[0]} });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 5, 5 },
        .{ 5, 5 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
