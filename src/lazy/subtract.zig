const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const subtractBackward = @import("../eager/subtract.zig").subtractBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const broadcastShape = @import("broadcast.zig").broadcastShape;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;

const Subtract = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Subtract, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.subtract(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.subtract(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.subtract(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.subtract(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.subtract(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.subtract(i8, context.allocator, tensor, y.i8) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 2);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try subtractBackward(f64, EagerBackwardContext(f64){
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
            const gradients = try subtractBackward(f32, EagerBackwardContext(f32){
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
            const gradients = try subtractBackward(f16, EagerBackwardContext(f16){
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

pub fn subtract(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    if (x.scalarType != y.scalarType)
        return error.ScalarTypeMismatch;
    const shape = try broadcastShape(&graph.arena.allocator, x, y);
    var subtract_operation = try graph.arena.allocator.create(Subtract);
    subtract_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&subtract_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = shape,
        .scalarType = x.scalarType,
    };
}

test "subtract scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, 5);
    const y = try constant(f64, &graph, 10);
    const z = try subtract(&graph, x, y);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual.f64, expected);
    std.testing.expectEqual(z.shape, &[_]usize{});
}

test "subtract matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(f64, &graph, .{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    const z = try subtract(&graph, x, y);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f64, actual.f64, expected);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 2 }));
}

test "subtract matrix i32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(i32, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(i32, &graph, .{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    const z = try subtract(&graph, x, y);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(i32, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(i32, actual.i32, expected);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 2 }));
}

test "subtract broadcast scalar rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f16, &graph, 3);
    const b = try constant(f16, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const c = try subtract(&graph, a, b);
    const d = try subtract(&graph, b, a);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ c, d });
    const expected = try eager.constant(f16, &arena.allocator, .{
        .{ 2, 5 },
        .{ 0, 7 },
        .{ 8, -3 },
    });
    const expected2 = try eager.constant(f16, &arena.allocator, .{
        .{ -2, -5 },
        .{ 0, -7 },
        .{ -8, 3 },
    });
    expectEqual(f16, actual[0].f16, expected);
    expectEqual(f16, actual[1].f16, expected2);
}

test "subtract broadcast rank 3 to rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(i64, &graph, .{
        .{
            .{ 1, 2 },
        },
        .{
            .{ 3, 4 },
        },
        .{
            .{ 5, 6 },
        },
    });
    const b = try constant(i64, &graph, .{
        .{.{
            .{ 1, 2 },
            .{ 3, 4 },
            .{ 5, 6 },
        }},
        .{.{
            .{ 7, 8 },
            .{ 9, 10 },
            .{ 11, 12 },
        }},
    });
    const c = try subtract(&graph, a, b);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(c);
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{
            .{
                .{ 0, 0 },
                .{ -2, -2 },
                .{ -4, -4 },
            },
            .{
                .{ 2, 2 },
                .{ 0, 0 },
                .{ -2, -2 },
            },
            .{
                .{ 4, 4 },
                .{ 2, 2 },
                .{ 0, 0 },
            },
        },
        .{
            .{
                .{ -6, -6 },
                .{ -8, -8 },
                .{ -10, -10 },
            },
            .{
                .{ -4, -4 },
                .{ -6, -6 },
                .{ -8, -8 },
            },
            .{
                .{ -2, -2 },
                .{ -4, -4 },
                .{ -6, -6 },
            },
        },
    });
    expectEqual(i64, actual.i64, expected);
}

test "gradient subtract" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try constant(f64, &graph, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const c = try subtract(&graph, a, b);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_a_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    const expected_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ -0.25, -0.25 },
        .{ -0.25, -0.25 },
    });
    std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 2 }));
    expectEqual(f64, actual[0].f64, expected_a_gradient);
    expectEqual(f64, actual[1].f64, expected_b_gradient);
}

test "subtract backwards broadcast rank 3 to rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, -5);
    const b = try constant(f64, &graph, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const c = try subtract(&graph, a, b);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_scalar_gradient = try eager.constant(f64, &arena.allocator, 1);
    const expected_tensor_gradient = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ -0.125, -0.125 },
            .{ -0.125, -0.125 },
        },
        .{
            .{ -0.125, -0.125 },
            .{ -0.125, -0.125 },
        },
    });
    expectEqual(f64, expected_scalar_gradient, actual[0].f64);
    expectEqual(f64, expected_tensor_gradient, actual[1].f64);
}

test "subtract backwards broadcast rank 3 to rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 2 },
        },
        .{
            .{ 3, 4 },
        },
        .{
            .{ 5, 6 },
        },
    });
    const b = try constant(f64, &graph, .{
        .{.{
            .{ 1, 2 },
            .{ 3, 4 },
            .{ 5, 6 },
        }},
        .{.{
            .{ 7, 8 },
            .{ 9, 10 },
            .{ 11, 12 },
        }},
    });
    const c = try subtract(&graph, a, b);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_rank_3_gradient = try eager.constant(f64, &arena.allocator, .{
        .{.{ 0.16667, 0.16667 }},
        .{.{ 0.16667, 0.16667 }},
        .{.{ 0.16667, 0.16667 }},
    });
    const expected_rank_4_gradient = try eager.constant(f64, &arena.allocator, .{
        .{.{
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
        }},
        .{.{
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
            .{ -0.0833, -0.0833 },
        }},
    });
    expectEqual(f64, expected_rank_3_gradient, actual[0].f64);
    expectEqual(f64, expected_rank_4_gradient, actual[1].f64);
}
