const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const naturalLogarithmBackward = @import("../eager/natural_logarithm.zig").naturalLogarithmBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;

const NaturalLogarithm = struct {
    operation: Operation,
    inputs: [1]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(NaturalLogarithm, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{ .f64 = try eager.naturalLogarithm(f64, context.allocator, tensor) },
        .f32 => |tensor| .{ .f32 = try eager.naturalLogarithm(f32, context.allocator, tensor) },
        else => return error.OperationNotDefinedForScalarType,
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try naturalLogarithmBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
                .forward_output = context.forward_output.f64,
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try naturalLogarithmBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
                .forward_output = context.forward_output.f32,
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => return error.OperationNotDefinedForScalarType,
        .i64, .i32, .i8 => return error.CannotDifferentiateIntegral,
    }
    return values;
}

pub fn naturalLogarithm(graph: *Graph, x: Tensor) !Tensor {
    var naturalLogarithm_operation = try graph.arena.allocator.create(NaturalLogarithm);
    naturalLogarithm_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
    };
    try graph.operations.append(&naturalLogarithm_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = x.shape,
        .scalarType = x.scalarType,
    };
}

test "naturalLogarithm scalar" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, 5);
    const y = try naturalLogarithm(&graph, x);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f64, &arena.allocator, 1.6094);
    expectEqual(f64, actual[0].f64, expected);
}

test "naturalLogarithm matrix" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const y = try naturalLogarithm(&graph, x);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{y}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0, 0.6931 },
        .{ 1.0986, 1.3862 },
        .{ 1.6094, 1.7917 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient naturalLogarithm" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const b = try naturalLogarithm(&graph, a);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 3, 2 }));
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.1666, 0.0833 },
        .{ 0.0555, 0.0416 },
        .{ 0.0333, 0.0277 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
