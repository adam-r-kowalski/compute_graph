const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const exponentiateBackward = @import("../eager/exponentiate.zig").exponentiateBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;

const Exponentiate = struct {
    operation: Operation,
    inputs: [1]Tensor,
};

fn inputs(operation: *const Operation) []const Tensor {
    return &@fieldParentPtr(Exponentiate, "operation", operation).inputs;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    return switch (context.values[0]) {
        .f64 => |tensor| .{ .f64 = try eager.exponentiate(f64, context.allocator, tensor) },
        .f32 => |tensor| .{ .f32 = try eager.exponentiate(f32, context.allocator, tensor) },
        else => return error.OperationNotDefinedForScalarType,
    };
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const gradients = try exponentiateBackward(f64, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const gradients = try exponentiateBackward(f32, EagerBackwardContext(f32){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f32){context.forward_inputs[0].f32},
            });
            values[0] = .{ .f32 = gradients[0] };
        },
        .f16 => return error.OperationNotDefinedForScalarType,
        .i64, .i32, .i8 => return error.CannotDifferentiateIntegral,
    }
    return values;
}

pub fn exponentiate(graph: *Graph, x: Tensor) !Tensor {
    var exponentiate_operation = try graph.arena.allocator.create(Exponentiate);
    exponentiate_operation.* = .{
        .operation = .{
            .inputs = inputs,
            .forward = forward,
            .backward = backward,
        },
        .inputs = .{x},
    };
    try graph.operations.append(&exponentiate_operation.operation);
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = x.shape,
    };
}

test "exponentiate scalar" {
    const constant = @import("constant.zig").constant;
    const Session = @import("session.zig").Session;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, @as(f64, -5));
    const y = try exponentiate(&graph, x);
    std.testing.expectEqual(y.shape, &[_]usize{});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, @as(f64, 0.00673));
    expectEqual(f64, actual[0].f64, expected);
}

test "exponentiate matrix" {
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
    const y = try exponentiate(&graph, x);
    std.testing.expect(std.mem.eql(usize, y.shape, &[_]usize{ 3, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{y} });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2.7182, 0.1353 },
        .{ 20.0855, 0.0183 },
        .{ 0.00673, 403.4287 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "gradient exponentiate" {
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
    const b = try exponentiate(&graph, a);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = gradients });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 0.6795, 1.8472 },
        .{ 5.0213, 13.6495 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
