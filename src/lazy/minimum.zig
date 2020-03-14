const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const ScalarType = tensor.ScalarType;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const minimumBackward = @import("../eager/minimum.zig").minimumBackward;
const newShape = @import("../eager/reduce.zig").newShape;
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;
const ReduceParameters = @import("../eager/reduce.zig").ReduceParameters;

const Minimum = struct {
    operation: Operation,
    inputs: [1]Tensor,
    parameters: ReduceParameters,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Minimum, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    const parameters = @fieldParentPtr(Minimum, "operation", context.operation).parameters;
    return switch (context.values[0]) {
        .f64 => |t| .{ .f64 = try eager.minimum(f64, context.allocator, t, parameters) },
        .f32 => |t| .{ .f32 = try eager.minimum(f32, context.allocator, t, parameters) },
        .f16 => |t| .{ .f16 = try eager.minimum(f16, context.allocator, t, parameters) },
        .i64 => |t| .{ .i64 = try eager.minimum(i64, context.allocator, t, parameters) },
        .i32 => |t| .{ .i32 = try eager.minimum(i32, context.allocator, t, parameters) },
        .i8 => |t| .{ .i8 = try eager.minimum(i8, context.allocator, t, parameters) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const parameters = @fieldParentPtr(Minimum, "operation", context.operation).parameters;
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try minimumBackward(f64, parameters, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
                .forward_output = context.forward_output.f64,
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try minimumBackward(f32, parameters, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
                .forward_output = context.forward_output.f32,
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => |gradient_input| {
            const gradients = try minimumBackward(f16, parameters, EagerBackwardContext(f16){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f16){context.forward_inputs[0].f16},
                .forward_output = context.forward_output.f16,
            });
            values[0] = .{ .f16 = gradients[0] };
        },
        .i64, .i32, .i8 => {
            return error.CannotDifferentiateIntegral;
        },
    }
    return values;
}

pub fn minimum(graph: *Graph, x: Tensor, parameters: ReduceParameters) !Tensor {
    var allocator = &graph.arena.allocator;
    var minimum_operation = try allocator.create(Minimum);
    minimum_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
        .parameters = parameters,
    };
    try graph.operations.append(&minimum_operation.operation);
    const shape = try newShape(allocator, x.shape, parameters);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = shape,
        .scalarType = x.scalarType,
    };
}

test "minimum rank 0" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, -5);
    const y = try minimum(&graph, x, .{});
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual[0].f64, expected);
}

test "minimum rank 1" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(i32, &graph, .{ 5, 10, 7, 8, 10 });
    const y = try minimum(&graph, x, .{});
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i32, &arena.allocator, 5);
    expectEqual(i32, actual[0].i32, expected);
}

test "minimum rank 2" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f16, &graph, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const y = try minimum(&graph, x, .{});
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f16, &arena.allocator, 5);
    expectEqual(f16, actual[0].f16, expected);
}

test "minimum rank 2 across 0 dimension" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f16, &graph, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const y = try minimum(&graph, x, .{ .dimension = 0 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{2}));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f16, &arena.allocator, .{ -3, 2 });
    expectEqual(f16, actual[0].f16, expected);
}

test "minimum rank 3" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(i8, &graph, .{
        .{
            .{ 5, 10 },
            .{ 7, 8 },
        },
        .{
            .{ 10, 8 },
            .{ 2, 6 },
        },
    });
    const y = try minimum(&graph, x, .{});
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{}));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i8, &arena.allocator, 2);
    expectEqual(i8, actual[0].i8, expected);
}

test "minimum rank 3 accross 0 dimension" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(i64, &graph, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const y = try minimum(&graph, x, .{ .dimension = 0 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
    });
    expectEqual(i64, actual[0].i64, expected);
}

test "minimum rank 3 accross 1 dimension" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(i64, &graph, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const y = try minimum(&graph, x, .{ .dimension = 1 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{ -3, 2 },
        .{ 5, 6 },
    });
    expectEqual(i64, actual[0].i64, expected);
}

test "minimum rank 3 accross 2 dimension" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(i64, &graph, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const y = try minimum(&graph, x, .{ .dimension = 2 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{ 1, -3 },
        .{ 5, 7 },
    });
    expectEqual(i64, actual[0].i64, expected);
}

test "gradient minimum rank 0" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, 4);
    const b = try minimum(&graph, a, .{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient minimum rank 1" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{ 1, 2, 3, 4, 5 });
    const b = try minimum(&graph, a, .{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{ 1, 0, 0, 0, 0 });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient minimum rank 1 repeated min" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{ 1, 1, 3, 4, 5 });
    const b = try minimum(&graph, a, .{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{ 0.5, 0.5, 0, 0, 0 });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient minimum rank 1 thrice repeated min" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{ 1, 1, 1, 4, 5 });
    const b = try minimum(&graph, a, .{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{ 0.3333, 0.3333, 0.3333, 0, 0 });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient minimum rank 2" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try minimum(&graph, a, .{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 1, 0 },
        .{ 0, 0 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient minimum rank 3 dimension 0" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try minimum(&graph, a, .{ .dimension = 0 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0.25 },
        },
        .{
            .{ 0, 0.25 },
            .{ 0, 0 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient minimum rank 3 dimension 1" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try minimum(&graph, a, .{ .dimension = 1 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0, 0 },
            .{ 0.25, 0.25 },
        },
        .{
            .{ 0.25, 0.25 },
            .{ 0, 0 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "minimum backward rank 3 dimension 2" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 12 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try minimum(&graph, a, .{ .dimension = 2 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0 },
        },
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "minimum backward rank 3 dimension 2 repeating min" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 1 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try minimum(&graph, a, .{ .dimension = 2 });
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.125, 0.125 },
            .{ 0.25, 0 },
        },
        .{
            .{ 0.25, 0 },
            .{ 0.25, 0 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}
