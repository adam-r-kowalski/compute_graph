const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const addBackward = @import("../eager/add.zig").addBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const broadcastShape = @import("broadcast.zig").broadcastShape;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;

const Add = struct {
    operation: Operation,
    inputs: [2]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Add, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 2);
    const x = context.values[0];
    const y = context.values[1];
    return switch (x) {
        .f64 => |tensor| .{ .f64 = try eager.add(f64, context.allocator, tensor, y.f64) },
        .f32 => |tensor| .{ .f32 = try eager.add(f32, context.allocator, tensor, y.f32) },
        .f16 => |tensor| .{ .f16 = try eager.add(f16, context.allocator, tensor, y.f16) },
        .i64 => |tensor| .{ .i64 = try eager.add(i64, context.allocator, tensor, y.i64) },
        .i32 => |tensor| .{ .i32 = try eager.add(i32, context.allocator, tensor, y.i32) },
        .i8 => |tensor| .{ .i8 = try eager.add(i8, context.allocator, tensor, y.i8) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 2);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try addBackward(f64, EagerBackwardContext(f64){
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
            const gradients = try addBackward(f32, EagerBackwardContext(f32){
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
            const gradients = try addBackward(f16, EagerBackwardContext(f16){
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

pub fn add(graph: *Graph, x: Tensor, y: Tensor) !Tensor {
    if (x.scalarType != y.scalarType)
        return error.ScalarTypeMismatch;
    const shape = try broadcastShape(&graph.arena.allocator, x, y);
    var add_operation = try graph.arena.allocator.create(Add);
    add_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{ x, y },
    };
    try graph.operations.append(&add_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = shape,
        .scalarType = x.scalarType,
    };
}

test "add scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, 5);
    const y = try constant(f64, &graph, 10);
    const z = try add(&graph, x, y);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(f64, &arena.allocator, 15);
    expectEqual(f64, actual.f64, expected);
    std.testing.expectEqual(z.shape, &[_]usize{});
    std.testing.expectEqual(z.scalarType, .f64);
}

test "add matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const z = try add(&graph, x, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f64, actual.f64, expected);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 2 }));
    std.testing.expectEqual(z.scalarType, .f64);
}

test "add matrix i32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(i32, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const z = try add(&graph, x, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(i32, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(i32, actual.i32, expected);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 3, 2 }));
    std.testing.expectEqual(z.scalarType, .i32);
}

test "add broadcast scalar rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(i8, &graph, -5);
    const y = try constant(i8, &graph, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const z = try add(&graph, x, y);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(i8, &arena.allocator, .{
        .{
            .{ -4, -7 },
            .{ -2, -9 },
        },
        .{
            .{ 0, -11 },
            .{ 2, -13 },
        },
    });
    expectEqual(i8, actual.i8, expected);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 2, 2, 2 }));
    std.testing.expectEqual(z.scalarType, .i8);
}

test "add broadcast rank 3 to rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(i64, &graph, .{
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
    const y = try constant(i64, &graph, .{
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
    const z = try add(&graph, x, y);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(z);
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{
            .{
                .{ 2, 4 },
                .{ 4, 6 },
                .{ 6, 8 },
            },
            .{
                .{ 4, 6 },
                .{ 6, 8 },
                .{ 8, 10 },
            },
            .{
                .{ 6, 8 },
                .{ 8, 10 },
                .{ 10, 12 },
            },
        },
        .{
            .{
                .{ 8, 10 },
                .{ 10, 12 },
                .{ 12, 14 },
            },
            .{
                .{ 10, 12 },
                .{ 12, 14 },
                .{ 14, 16 },
            },
            .{
                .{ 12, 14 },
                .{ 14, 16 },
                .{ 16, 18 },
            },
        },
    });
    expectEqual(i64, actual.i64, expected);
    std.testing.expect(std.mem.eql(usize, z.shape, &[_]usize{ 2, 3, 3, 2 }));
    std.testing.expectEqual(z.scalarType, .i64);
}

test "gradient add" {
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
    const c = try add(&graph, a, b);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.25, 0.25 },
        .{ 0.25, 0.25 },
    });
    expectEqual(f64, actual[0].f64, expected);
    expectEqual(f64, actual[1].f64, expected);
    std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 2 }));
    std.testing.expectEqual(c.scalarType, .f64);
    std.testing.expect(std.mem.eql(usize, gradients[0].shape, &[_]usize{ 2, 2 }));
    std.testing.expect(std.mem.eql(usize, gradients[1].shape, &[_]usize{ 2, 2 }));
}

test "gradient add broadcast scalar rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f32, &graph, -5);
    const b = try constant(f32, &graph, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const c = try add(&graph, a, b);
    const d = try mean(&graph, c);
    const gradients = try gradient(&graph, d, &[_]Tensor{ a, b });
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected_a_gradient = try eager.constant(f32, &arena.allocator, 1);
    const expected_b_gradient = try eager.constant(f32, &arena.allocator, .{
        .{
            .{
                0.125,
                0.125,
            },
            .{
                0.125,
                0.125,
            },
        },
        .{
            .{
                0.125,
                0.125,
            },
            .{
                0.125,
                0.125,
            },
        },
    });
    expectEqual(f32, actual[0].f32, expected_a_gradient);
    expectEqual(f32, actual[1].f32, expected_b_gradient);
    std.testing.expect(std.mem.eql(usize, gradients[0].shape, &[_]usize{}));
    std.testing.expect(std.mem.eql(usize, gradients[1].shape, &[_]usize{ 2, 2, 2 }));
    std.testing.expectEqual(gradients[0].scalarType, .f32);
    std.testing.expectEqual(gradients[1].scalarType, .f32);
}

test "add matrix shape mismatch" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
    });
    const y = try constant(f64, &graph, .{
        .{ 1, -2, 3 },
        .{ 3, -4, 5 },
        .{ 3, -4, 6 },
    });
    _ = add(&graph, x, y) catch |err| switch (err) {
        error.CouldNotBroadcastShapes => {},
        else => unreachable,
    };
}

test "add matrix scalar type mismatch" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
    });
    const y = try constant(f32, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
    });
    _ = add(&graph, x, y) catch |err| switch (err) {
        error.ScalarTypeMismatch => {},
        else => unreachable,
    };
}
