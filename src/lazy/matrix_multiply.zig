const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const matrixMultiplyBackward = @import("../eager/matrix_multiply.zig").matrixMultiplyBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const MatrixMultiply = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(MatrixMultiply, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.matrixMultiply(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.matrixMultiply(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.matrixMultiply(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.matrixMultiply(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.matrixMultiply(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.matrixMultiply(i8, context.allocator, tensor, y.i8) },
    };
}
fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 2);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try matrixMultiplyBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){
                    context.forward_inputs[0].f64,
                    context.forward_inputs[1].f64,
                },
                .forward_output = context.forward_output.f64,
            });
            values[0] = .{ .f64 = gradients[0] };
            values[1] = .{ .f64 = gradients[1] };
        },
        .f32 => |gradient_input| {
            const gradients = try matrixMultiplyBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){
                    context.forward_inputs[0].f32,
                    context.forward_inputs[1].f32,
                },
                .forward_output = context.forward_output.f32,
            });
            values[0] = .{ .f32 = gradients[0] };
            values[1] = .{ .f32 = gradients[1] };
        },
        .f16 => |gradient_input| {
            const gradients = try matrixMultiplyBackward(f16, EagerBackwardContext(f16){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f16){
                    context.forward_inputs[0].f16,
                    context.forward_inputs[1].f16,
                },
                .forward_output = context.forward_output.f16,
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

pub fn matrixMultiply(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    if (x.shape.len != 2 or y.shape.len != 2 or x.shape[1] != y.shape[0])
        return error.ShapeMismatch;
    if (x.scalarType != y.scalarType)
        return error.ScalarTypeMismatch;
    var shape = try graph.arena.allocator.alloc(usize, 2);
    shape[0] = x.shape[0];
    shape[1] = y.shape[1];
    var matrixMultiply_operation = try graph.arena.allocator.create(MatrixMultiply);
    matrixMultiply_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&matrixMultiply_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = shape,
        .scalarType = x.scalarType,
    };
}

test "matrixMultiply identity" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, .{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    });
    const y = try constant(f64, &graph, .{
        .{1},
        .{2},
        .{3},
    });
    const z = try matrixMultiply(&graph, x, y);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 1 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{1},
        .{2},
        .{3},
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "matrixMultiply flip" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, .{
        .{ 1, 0, 0 },
        .{ 0, -1, 0 },
        .{ 0, 0, 1 },
    });
    const y = try constant(f64, &graph, .{
        .{1},
        .{2},
        .{3},
    });
    const z = try matrixMultiply(&graph, x, y);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 1 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{1},
        .{-2},
        .{3},
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "matrixMultiply flip" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, .{
        .{ 1, 2, 3 },
        .{ 4, 2, 5 },
        .{ 9, 8, 4 },
        .{ 6, 5, 3 },
    });
    const y = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 4, 6 },
        .{ 3, 9 },
    });
    const z = try matrixMultiply(&graph, x, y);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 4, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 18, 41 },
        .{ 27, 65 },
        .{ 53, 102 },
        .{ 35, 69 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient matrix multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const b = try constant(f64, &graph, .{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const c = try matrixMultiply(&graph, a, b);
    std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 2 }));
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected_a_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ 3.75, 4.75, 5.75 },
        .{ 3.75, 4.75, 5.75 },
    });
    const expected_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ 1.25, 1.25 },
        .{ 1.75, 1.75 },
        .{ 2.25, 2.25 },
    });
    expectEqual(f64, actual[0].f64, expected_a_gradient);
    expectEqual(f64, actual[1].f64, expected_b_gradient);
}

test "matrixMultiply shape mismatch" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, .{
        .{ 1, 2, 3 },
        .{ 4, 2, 5 },
        .{ 6, 5, 3 },
    });
    const y = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 4, 6 },
    });
    _ = matrixMultiply(&graph, x, y) catch |err| switch (err) {
        error.ShapeMismatch => {},
        else => unreachable,
    };
}
