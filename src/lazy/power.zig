const std = @import("std");
const Allocator = std.mem.Allocator;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Operation = @import("operation.zig").Operation;
const eager = @import("../eager.zig");
const CpuTensor = eager.CpuTensor;
const CpuTensorUnion = eager.CpuTensorUnion;
const expectEqual = @import("../testing.zig").expectEqual;
const powerBackward = @import("../eager/power.zig").powerBackward;
const EagerBackwardContext = @import("../eager/backward.zig").Context;
const constant = @import("constant.zig").constant;
const Session = @import("session.zig").Session;
const gradient = @import("gradient.zig").gradient;
const mean = @import("mean.zig").mean;

fn Power(comptime T: type) type {
    return struct {
        operation: Operation,
        inputs: [1]Tensor,
        n: T,
    };
}

fn inputs(comptime T: type) fn (*const Operation) []const Tensor {
    return struct {
        fn call(operation: *const Operation) []const Tensor {
            return &@fieldParentPtr(Power(f64), "operation", operation).inputs;
        }
    }.call;
}

fn forward(context: Operation.ForwardContext) Operation.ForwardResult {
    std.debug.assert(context.values.len == 1);
    switch (context.values[0]) {
        .f64 => |tensor| {
            const n = @fieldParentPtr(Power(f64), "operation", context.operation).n;
            return CpuTensorUnion{ .f64 = try eager.power(f64, context.allocator, tensor, n) };
        },
        .f32 => |tensor| {
            const n = @fieldParentPtr(Power(f32), "operation", context.operation).n;
            return CpuTensorUnion{ .f32 = try eager.power(f32, context.allocator, tensor, n) };
        },
        .i64 => |tensor| {
            const n = @fieldParentPtr(Power(i64), "operation", context.operation).n;
            return CpuTensorUnion{ .i64 = try eager.power(i64, context.allocator, tensor, n) };
        },
        .i32 => |tensor| {
            const n = @fieldParentPtr(Power(i32), "operation", context.operation).n;
            return CpuTensorUnion{ .i32 = try eager.power(i32, context.allocator, tensor, n) };
        },
        .i8 => |tensor| {
            const n = @fieldParentPtr(Power(i8), "operation", context.operation).n;
            return CpuTensorUnion{ .i8 = try eager.power(i8, context.allocator, tensor, n) };
        },
        else => return error.OperationNotDefinedForScalarType,
    }
}

fn backward(context: Operation.BackwardContext) Operation.BackwardResult {
    const values = try context.allocator.alloc(CpuTensorUnion, 1);
    errdefer context.allocator.free(values);
    switch (context.gradient_input) {
        .f64 => |gradient_input| {
            const n = @fieldParentPtr(Power(f64), "operation", context.operation).n;
            const gradients = try powerBackward(f64, n, EagerBackwardContext(f64){
                .allocator = context.allocator,
                .gradient_input = gradient_input,
                .forward_inputs = &[_]CpuTensor(f64){context.forward_inputs[0].f64},
                .forward_output = context.forward_output.f64,
            });
            values[0] = .{ .f64 = gradients[0] };
        },
        .f32 => |gradient_input| {
            const n = @fieldParentPtr(Power(f32), "operation", context.operation).n;
            const gradients = try powerBackward(f32, n, EagerBackwardContext(f32){
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

fn isInt(comptime T: type) bool {
    return std.meta.trait.isSignedInt(T) or std.meta.trait.isUnsignedInt(T);
}

pub fn power(graph: *Graph, x: Tensor, n: var) !Tensor {
    switch (x.scalarType) {
        .f64 => {
            var power_operation = try graph.arena.allocator.create(Power(f64));
            power_operation.* = .{
                .operation = .{
                    .inputs = inputs(f64),
                    .forward = forward,
                    .backward = backward,
                },
                .inputs = .{x},
                .n = n,
            };
            try graph.operations.append(&power_operation.operation);
        },
        .f32 => {
            var power_operation = try graph.arena.allocator.create(Power(f32));
            power_operation.* = .{
                .operation = .{
                    .inputs = inputs(f32),
                    .forward = forward,
                    .backward = backward,
                },
                .inputs = .{x},
                .n = n,
            };
            try graph.operations.append(&power_operation.operation);
        },
        .f16 => {
            var power_operation = try graph.arena.allocator.create(Power(f16));
            power_operation.* = .{
                .operation = .{
                    .inputs = inputs(f16),
                    .forward = forward,
                    .backward = backward,
                },
                .inputs = .{x},
                .n = n,
            };
            try graph.operations.append(&power_operation.operation);
        },
        .i64 => {
            if (comptime !isInt(@TypeOf(n))) {
                return error.CannotRaiseIntToFloatPower;
            } else {
                var power_operation = try graph.arena.allocator.create(Power(i64));
                power_operation.* = .{
                    .operation = .{
                        .inputs = inputs(i64),
                        .forward = forward,
                        .backward = backward,
                    },
                    .inputs = .{x},
                    .n = n,
                };
                try graph.operations.append(&power_operation.operation);
            }
        },
        .i32 => {
            if (comptime !isInt(@TypeOf(n))) {
                return error.CannotRaiseIntToFloatPower;
            } else {
                var power_operation = try graph.arena.allocator.create(Power(i32));
                power_operation.* = .{
                    .operation = .{
                        .inputs = inputs(i32),
                        .forward = forward,
                        .backward = backward,
                    },
                    .inputs = .{x},
                    .n = n,
                };
                try graph.operations.append(&power_operation.operation);
            }
        },
        .i8 => {
            if (comptime !isInt(@TypeOf(n))) {
                return error.CannotRaiseIntToFloatPower;
            } else {
                var power_operation = try graph.arena.allocator.create(Power(i8));
                power_operation.* = .{
                    .operation = .{
                        .inputs = inputs(i8),
                        .forward = forward,
                        .backward = backward,
                    },
                    .inputs = .{x},
                    .n = n,
                };
                try graph.operations.append(&power_operation.operation);
            }
        },
    }
    return Tensor{
        .tensorType = .{ .operation = graph.operations.len - 1 },
        .shape = x.shape,
        .scalarType = x.scalarType,
    };
}

test "power rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, -5);
    const y = try power(&graph, x, 2.0);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{y});
    const expected = try eager.constant(f64, &arena.allocator, 25);
    expectEqual(f64, actual[0].f64, expected);
}

test "power rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(i32, &graph, .{ 1, -2, 3, -4, -5, 6 });
    const y = try power(&graph, x, 3);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{y});
    const expected = try eager.constant(i32, &arena.allocator, .{ 1, -8, 27, -64, -125, 216 });
    expectEqual(i32, actual[0].i32, expected);
}

test "power rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f32, &graph, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const y = try power(&graph, x, -2);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{y});
    const expected = try eager.constant(f32, &arena.allocator, .{
        .{ 1, 0.25 },
        .{ 0.1111, 0.0625 },
        .{ 0.04, 0.0277 },
    });
    expectEqual(f32, actual[0].f32, expected);
}

test "power rank 2 float" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f32, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const y = try power(&graph, x, 2.5);
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{y});
    const expected = try eager.constant(f32, &arena.allocator, .{
        .{ 1, 5.6568 },
        .{ 15.5884, 32 },
        .{ 55.9016, 88.1816 },
    });
    expectEqual(f32, actual[0].f32, expected);
}

test "gradient power rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try power(&graph, a, 2.5);
    const c = try mean(&graph, b);
    const gradients = try gradient(&graph, c, &[_]Tensor{a});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 0.625, 1.7677 },
        .{ 3.2475, 5 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
