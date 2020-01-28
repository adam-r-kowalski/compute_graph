const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const multiplyBackward = @import("../eager/multiply.zig").multiplyBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const Multiply = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Multiply, "operation", operation).inputs;
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

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 2);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try multiplyBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){
                    context.forward_inputs[0].f64,
                    context.forward_inputs[1].f64,
                },
            });
            values[0] = .{ .f64 = gradients[0] };
            values[1] = .{ .f64 = gradients[1] };
        },
        .f32 => |gradient_input| {
            const gradients = try multiplyBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){
                    context.forward_inputs[0].f32,
                    context.forward_inputs[1].f32,
                },
            });
            values[0] = .{ .f32 = gradients[0] };
            values[1] = .{ .f32 = gradients[1] };
        },
        .f16 => |gradient_input| {
            const gradients = try multiplyBackward(f16, EagerBackwardContext(f16){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f16){
                    context.forward_inputs[0].f16,
                    context.forward_inputs[1].f16,
                },
            });
            values[0] = .{ .f16 = gradients[0] };
            values[1] = .{ .f16 = gradients[1] };
        },
        .i64, .i32, .i8 => {
            return error.CannotDifferentiateIntegral;
        },
    }
    return values;
}

pub fn multiply(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    var multiply_operation = try graph.arena.allocator.create(Multiply);
    multiply_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&multiply_operation.operation);
    return Tensor{ .operation = graph.operations.len - 1 };
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
    const actual = try session.run(&[_]Tensor{z});
    const expected = try eager.constant(&arena.allocator, @as(f64, 50));
    expectEqual(f64, actual[0].f64, expected);
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
    const actual = try session.run(&[_]Tensor{z});
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 4 },
        .{ 9, 16 },
        .{ 25, 36 },
    });
    expectEqual(f64, actual[0].f64, expected);
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
    const actual = try session.run(&[_]Tensor{z});
    const expected = try eager.constant(&arena.allocator, [_][2]i32{
        .{ 1, 4 },
        .{ 9, 16 },
        .{ 25, 36 },
    });
    expectEqual(i32, actual[0].i32, expected);
}

test "gradient multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try constant(&graph, [_][2]f64{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const c = try multiply(&graph, a, b);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients);
    const expected_a_gradient = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 5 * 0.25, 6 * 0.25 },
        .{ 7 * 0.25, 8 * 0.25 },
    });
    const expected_b_gradient = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1 * 0.25, 2 * 0.25 },
        .{ 3 * 0.25, 4 * 0.25 },
    });
    expectEqual(f64, actual[0].f64, expected_a_gradient);
    expectEqual(f64, actual[1].f64, expected_b_gradient);
}
