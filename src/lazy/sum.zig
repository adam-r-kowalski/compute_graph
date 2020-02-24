const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const ScalarType = tensor.ScalarType;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const eager_sum = @import("../eager/sum.zig");
const sumBackward = eager_sum.sumBackward;
const newShape = eager_sum.newShape;
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const Sum = struct {
    operation: Operation,
    inputs: [1]Tensor,
    dimension: ?usize,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Sum, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    const dimension = @fieldParentPtr(Sum, "operation", context.operation).dimension;
    return switch (context.values[0]) {
        .f64 => |t| .{ .f64 = try eager.sum(f64, context.allocator, t, dimension) },
        .f32 => |t| .{ .f32 = try eager.sum(f32, context.allocator, t, dimension) },
        .f16 => |t| .{ .f16 = try eager.sum(f16, context.allocator, t, dimension) },
        .i64 => |t| .{ .i64 = try eager.sum(i64, context.allocator, t, dimension) },
        .i32 => |t| .{ .i32 = try eager.sum(i32, context.allocator, t, dimension) },
        .i8 => |t| .{ .i8 = try eager.sum(i8, context.allocator, t, dimension) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const dimension = @fieldParentPtr(Sum, "operation", context.operation).dimension;
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try sumBackward(f64, dimension, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try sumBackward(f32, dimension, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => |gradient_input| {
            const gradients = try sumBackward(f16, dimension, EagerBackwardContext(f16){
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

const SumParameters = struct {
    dimension: ?usize = null,
};

pub fn sum(graph: *Graph, x: Tensor, parameters: SumParameters) !Tensor {
    var allocator = &graph.arena.allocator;
    var sum_operation = try allocator.create(Sum);
    sum_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
        .dimension = parameters.dimension,
    };
    try graph.operations.append(&sum_operation.operation);
    const shape = try newShape(allocator, x.shape, parameters.dimension);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = shape,
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
    const x = try constant(f64, &graph, -5);
    const y = try sum(&graph, x, .{});
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f64, &arena.allocator, -5);
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
    const x = try constant(f64, &graph, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const y = try sum(&graph, x, .{});
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f64, &arena.allocator, 48);
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
    const x = try constant(i32, &graph, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const y = try sum(&graph, x, .{});
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i32, &arena.allocator, 48);
    expectEqual(i32, actual[0].i32, expected);
}

test "sum rank 3 accross 0 dimension" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
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
    const y = try sum(&graph, x, .{ .dimension = 0 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{ 6, 8 },
        .{ 4, 12 },
    });
    expectEqual(i64, actual[0].i64, expected);
}

test "sum rank 3 accross 1 dimension" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
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
    const y = try sum(&graph, x, .{ .dimension = 1 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{ -2, 6 },
        .{ 12, 14 },
    });
    expectEqual(i64, actual[0].i64, expected);
}

test "sum rank 3 accross 2 dimension" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
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
    const y = try sum(&graph, x, .{ .dimension = 2 });
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(i64, &arena.allocator, .{
        .{ 3, 1 },
        .{ 11, 15 },
    });
    expectEqual(i64, actual[0].i64, expected);
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
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try sum(&graph, a, .{});
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
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
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try sum(&graph, a, .{});
    const c = try constant(f64, &graph, 5);
    const d = try multiply(&graph, b, c);
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, d, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 5, 5 },
        .{ 5, 5 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sum rank 1 with multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const multiply = @import("multiply.zig").multiply;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{ 1, 2, 3, 4 });
    const b = try sum(&graph, a, .{});
    const c = try constant(f64, &graph, 5);
    const d = try multiply(&graph, b, c);
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, d, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{ 5, 5, 5, 5 });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sum rank 1 dimension 0 with multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const multiply = @import("multiply.zig").multiply;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{ 1, 2, 3, 4 });
    const b = try sum(&graph, a, .{ .dimension = 0 });
    const c = try constant(f64, &graph, 5);
    const d = try multiply(&graph, b, c);
    std.testing.expectEqual(b.shape, &[_]usize{});
    const gradients = try gradient(&graph, d, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{ 5, 5, 5, 5 });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sum rank 3 dimension 0 with multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const multiply = @import("multiply.zig").multiply;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try sum(&graph, a, .{ .dimension = 0 });
    const c = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const d = try multiply(&graph, b, c);
    const e = try mean(&graph, d);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
    const gradients = try gradient(&graph, e, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.5 },
            .{ 0.75, 1 },
        },
        .{
            .{ 0.25, 0.5 },
            .{ 0.75, 1 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sum rank 3 dimension 1 with multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const multiply = @import("multiply.zig").multiply;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try sum(&graph, a, .{ .dimension = 1 });
    const c = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const d = try multiply(&graph, b, c);
    const e = try mean(&graph, d);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
    const gradients = try gradient(&graph, e, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.5 },
            .{ 0.25, 0.5 },
        },
        .{
            .{ 0.75, 1 },
            .{ 0.75, 1 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient sum rank 3 dimension 2 with multiply" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const gradient = @import("gradient.zig").gradient;
    const multiply = @import("multiply.zig").multiply;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const b = try sum(&graph, a, .{ .dimension = 2 });
    const c = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const d = try multiply(&graph, b, c);
    const e = try mean(&graph, d);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
    const gradients = try gradient(&graph, e, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{gradients[0]}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{
            .{ 0.25, 0.25 },
            .{ 0.5, 0.5 },
        },
        .{
            .{ 0.75, 0.75 },
            .{ 1, 1 },
        },
    });
    expectEqual(f64, actual[0].f64, expected);
}
