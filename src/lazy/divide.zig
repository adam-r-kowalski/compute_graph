const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const divideBackward = @import("../eager/divide.zig").divideBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const broadcastShape = @import("broadcast.zig").broadcastShape;

const Divide = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Divide, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.divide(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.divide(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.divide(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.divide(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.divide(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.divide(i8, context.allocator, tensor, y.i8) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 2);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try divideBackward(f64, EagerBackwardContext(f64){
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
            const gradients = try divideBackward(f32, EagerBackwardContext(f32){
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
            const gradients = try divideBackward(f16, EagerBackwardContext(f16){
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

pub fn divide(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    if (x.scalarType != y.scalarType)
        return error.ScalarTypeMismatch;
    const shape = try broadcastShape(&graph.arena.allocator, x, y);
    var divide_operation = try graph.arena.allocator.create(Divide);
    divide_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&divide_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = shape,
        .scalarType = x.scalarType,
    };
}

test "divide scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, 5);
    const y = try constant(f64, &graph, 10);
    const z = try divide(&graph, x, y);
    std.testing.expectEqual(z.shape, &[_]usize{});
    std.testing.expectEqual(z.scalarType, .f64);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z}, .{});
    const expected = try eager.constant(f64, &arena.allocator, 0.5);
    expectEqual(f64, actual[0].f64, expected);
}

test "divide matrix" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(f64, &graph, .{
        .{ 6, -5 },
        .{ 4, -3 },
        .{ -2, 1 },
    });
    const z = try divide(&graph, x, y);
    std.testing.expectEqual(z.scalarType, .f64);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.1666, 0.4 },
        .{ 0.75, 1.3333 },
        .{ 2.5, 6 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "divide matrix i32" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(i32, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try constant(i32, &graph, .{
        .{ 6, -5 },
        .{ 4, -3 },
        .{ -2, 1 },
    });
    const z = try divide(&graph, x, y);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 2 }));
    std.testing.expectEqual(z.scalarType, .i32);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{z}, .{});
    const expected = try eager.constant(i32, &arena.allocator, .{
        .{ 0, 0 },
        .{ 0, 1 },
        .{ 2, 6 },
    });
    expectEqual(i32, actual[0].i32, expected);
}

test "gradient divide" {
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
        .{ 7, 8, 9 },
        .{ 10, 11, 12 },
    });
    const c = try divide(&graph, a, b);
    std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 3 }));
    std.testing.expectEqual(c.scalarType, .f64);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected_a_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ 0.0238, 0.02083, 0.0185 },
        .{ 0.0166, 0.0151, 0.0138 },
    });
    const expected_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ -0.0034, -0.0052, -0.0061 },
        .{ -0.0066, -0.0068, -0.0069 },
    });
    expectEqual(f64, actual[0].f64, expected_a_gradient);
    expectEqual(f64, actual[1].f64, expected_b_gradient);
}

test "gradient divide broadcast scalar rank 3" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
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
    const c = try divide(&graph, a, b);
    const d = try mean(&graph, c);
    const e = try divide(&graph, b, a);
    const f = try mean(&graph, e);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    const gradients2 = try gradient(&graph, f, &[_]Tensor{ a, b });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const actual2 = try session.run(gradients2, .{});
    const expected_a_gradient = try eager.constant(f64, &arena.allocator, 0.0793);
    const expected_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.625, 0.1562 },
            .{ 0.0694, 0.0391 },
        },
        .{
            .{ 0.025, 0.0174 },
            .{ 0.0128, 0.0098 },
        },
    });
    const expected2_a_gradient = try eager.constant(f64, &arena.allocator, 0.02);
    const expected2_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ -0.025, -0.025 },
            .{ -0.025, -0.025 },
        },
        .{
            .{ -0.025, -0.025 },
            .{ -0.025, -0.025 },
        },
    });
    expectEqual(f64, actual[0].f64, expected_a_gradient);
    expectEqual(f64, actual[1].f64, expected_b_gradient);
    expectEqual(f64, actual2[0].f64, expected2_a_gradient);
    expectEqual(f64, actual2[1].f64, expected2_b_gradient);
}

test "gradient divide broadcast rank 3 to rank 4" {
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
    const c = try divide(&graph, a, b);
    const d = try mean(&graph, c);
    const e = try divide(&graph, b, a);
    const f = try mean(&graph, e);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    const gradients2 = try gradient(&graph, f, &[_]Tensor{ a, b });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const actual2 = try session.run(gradients2, .{});
    const expected_a_gradient = try eager.constant(f64, &arena.allocator, .{
        .{.{ 0.0522, 0.0340 }},
        .{.{ 0.0522, 0.0340 }},
        .{.{ 0.0522, 0.0340 }},
    });
    const expected_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{.{
            .{ -0.25, -0.0833 },
            .{ -0.0278, -0.0208 },
            .{ -0.01, -0.0093 },
        }},
        .{.{
            .{ -0.0051, -0.0052 },
            .{ -0.0031, -0.0033 },
            .{ -0.0021, -0.0023 },
        }},
    });
    const expected2_a_gradient = try eager.constant(f64, &arena.allocator, .{
        .{.{ -1.0, -0.2917 }},
        .{.{ -0.1111, -0.0729 }},
        .{.{ -0.04, -0.0324 }},
    });
    const expected2_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{.{
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
        }},
        .{.{
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
            .{ 0.0426, 0.0255 },
        }},
    });
    expectEqual(f64, actual[0].f64, expected_a_gradient);
    expectEqual(f64, actual[1].f64, expected_b_gradient);
    expectEqual(f64, actual2[0].f64, expected2_a_gradient);
    expectEqual(f64, actual2[1].f64, expected2_b_gradient);
}
