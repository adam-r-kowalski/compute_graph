const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const negateBackward = @import("../eager/negate.zig").negateBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;

const Negate = struct {
    operation: Operation,
    inputs: [1]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Negate, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{ .f64 = try eager.negate(f64, context.allocator, tensor) },
        .f32 => |tensor| .{ .f32 = try eager.negate(f32, context.allocator, tensor) },
        .f16 => |tensor| .{ .f16 = try eager.negate(f16, context.allocator, tensor) },
        .i64 => |tensor| .{ .i64 = try eager.negate(i64, context.allocator, tensor) },
        .i32 => |tensor| .{ .i32 = try eager.negate(i32, context.allocator, tensor) },
        .i8 => |tensor| .{ .i8 = try eager.negate(i8, context.allocator, tensor) },
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try negateBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
                .forward_output = context.forward_output.f64,
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try negateBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
                .forward_output = context.forward_output.f32,
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => |gradient_input| {
            const gradients = try negateBackward(f16, EagerBackwardContext(f16){
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

pub fn negate(graph: *Graph, x: Tensor) !Tensor {
    var negate_operation = try graph.arena.allocator.create(Negate);
    negate_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
    };
    try graph.operations.append(&negate_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = x.shape,
        .scalarType = x.scalarType,
    };
}

test "negate scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, -5);
    const y = try negate(&graph, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(y);
    const expected = try eager.constant(f64, &arena.allocator, 5);
    expectEqual(f64, actual.f64, expected);
    std.testing.expectEqual(y.shape, &[_]usize{});
}

test "negate matrix" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try negate(&graph, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(y);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    expectEqual(f64, actual.f64, expected);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
}

test "negate matrix i32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(i32, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try negate(&graph, x);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(y);
    const expected = try eager.constant(i32, &arena.allocator, .{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    expectEqual(i32, actual.i32, expected);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
}

test "gradient negate" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 0, -2 },
        .{ 3, -4 },
    });
    const b = try negate(&graph, a);
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ -0.25, -0.25 },
        .{ -0.25, -0.25 },
    });
    expectEqual(f64, actual[0].f64, expected);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
}
