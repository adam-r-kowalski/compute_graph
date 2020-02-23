const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const ScalarType = tensor.ScalarType;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const meanBackward = @import("../eager/mean.zig").meanBackward;
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const Mean = struct {
    operation: Operation,
    inputs: [1]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Mean, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |t| .{ .f64 = try eager.mean(f64, context.allocator, t) },
        .f32 => |t| .{ .f32 = try eager.mean(f32, context.allocator, t) },
        .f16 => |t| .{ .f16 = try eager.mean(f16, context.allocator, t) },
        .i64 => |t| .{ .f64 = try eager.mean(i64, context.allocator, t) },
        .i32 => |t| .{ .f32 = try eager.mean(i32, context.allocator, t) },
        .i8 => |t| .{ .f16 = try eager.mean(i8, context.allocator, t) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try meanBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try meanBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => |gradient_input| {
            const gradients = try meanBackward(f16, EagerBackwardContext(f16){
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

fn tensorScalarType(scalarType: ScalarType) ScalarType {
    return switch (scalarType) {
        .f64 => .f64,
        .f32 => .f32,
        .f16 => .f16,
        .i64 => .f64,
        .i32 => .f32,
        .i8 => .f16,
    };
}

pub fn mean(graph: *Graph, x: Tensor) !Tensor {
    var mean_operation = try graph.arena.allocator.create(Mean);
    mean_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
    };
    try graph.operations.append(&mean_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = &[_]usize{},
        .scalarType = tensorScalarType(x.scalarType),
    };
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
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(&arena.allocator, @as(f64, -5));
    expectEqual(f64, actual[0].f64, expected);
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
    const y = try mean(&graph, x);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(&arena.allocator, @as(f64, 8));
    expectEqual(f64, actual[0].f64, expected);
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
    const y = try mean(&graph, x);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(&arena.allocator, @as(f32, 8));
    expectEqual(f32, actual[0].f32, expected);
}

test "gradient mean" {
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
    const b = try mean(&graph, a);
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
