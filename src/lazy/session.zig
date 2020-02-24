const std = @import("std");
const Allocator = std.mem.Allocator;
const gradient = @import("gradient.zig").gradient;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const GradientHandle = @import("tensor.zig").GradientHandle;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensorUnion = eager.CpuTensorUnion;

const Cache = std.AutoHashMap(Tensor, CpuTensorUnion);
const GradientCaches = std.AutoHashMap(usize, Cache);
pub const Environment = std.AutoHashMap(Tensor, Tensor);

fn getValue(comptime Map: type, comptime Key: type, comptime Value: type, map: Map, key: Key) !Value {
    if (map.getValue(key)) |value| return value;
    return error.KeyNotFound;
}

const ExecutionOrder = struct {
    const Tensors = std.ArrayList(Tensor);
    const Visited = std.AutoHashMap(Tensor, void);
    const Error = error{
        OutOfMemory,
        KeyNotFound,
    };

    fn recurse(execution_order: *Tensors, visited: *Visited, graph: *const Graph, environment: Environment, tensor: Tensor) Error!void {
        switch (tensor.tensorType) {
            .operation => |index| {
                const operation = graph.operations.at(index);
                for (operation.inputs(operation)) |input|
                    if (!visited.contains(input))
                        try recurse(execution_order, visited, graph, environment, input);
            },
            .gradient_handle => |gradient_handle| {
                const of = graph.gradients.at(gradient_handle.gradient).of;
                if (!visited.contains(of))
                    try recurse(execution_order, visited, graph, environment, of);
            },
            .variable => |index| {
                const variable = graph.variables.at(index);
                if (!visited.contains(variable.current_value))
                    try recurse(execution_order, visited, graph, environment, variable.current_value);
            },
            .assign => |index| {
                const assign = graph.assigns.at(index);
                if (!visited.contains(assign.variable))
                    try recurse(execution_order, visited, graph, environment, assign.variable);
                if (!visited.contains(assign.value))
                    try recurse(execution_order, visited, graph, environment, assign.value);
            },
            .placeholder => {
                const desired_tensor = try getValue(Environment, Tensor, Tensor, environment, tensor);
                if (!visited.contains(desired_tensor))
                    try recurse(execution_order, visited, graph, environment, desired_tensor);
            },
            else => {},
        }
        try visited.putNoClobber(tensor, undefined);
        try execution_order.append(tensor);
    }
};

fn executionOrder(session: Session, tensors: []const Tensor, environment: Environment) ![]const Tensor {
    var execution_order = ExecutionOrder.Tensors.init(&session.arena.allocator);
    errdefer execution_order.deinit();
    var visited = ExecutionOrder.Visited.init(session.arena.child_allocator);
    defer visited.deinit();
    for (tensors) |tensor|
        if (!visited.contains(tensor))
            try ExecutionOrder.recurse(&execution_order, &visited, session.graph, environment, tensor);
    return execution_order.toSlice();
}

test "execution order" {
    const constant = @import("constant.zig").constant;
    const add = @import("add.zig").add;
    const multiply = @import("multiply.zig").multiply;
    const subtract = @import("subtract.zig").subtract;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try constant(f64, &graph, 3);
    const b = try constant(f64, &graph, 5);
    const y = try constant(f64, &graph, 35);
    const x = try constant(f64, &graph, 10);
    const h = try multiply(&graph, m, x);
    const y_hat = try add(&graph, h, b);
    const loss = try subtract(&graph, y, y_hat);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(session, &[_]Tensor{loss}, environment);
    std.testing.expectEqual(execution_order.len, 7);
    std.testing.expectEqual(execution_order[0], y);
    std.testing.expectEqual(execution_order[1], m);
    std.testing.expectEqual(execution_order[2], x);
    std.testing.expectEqual(execution_order[3], h);
    std.testing.expectEqual(execution_order[4], b);
    std.testing.expectEqual(execution_order[5], y_hat);
    std.testing.expectEqual(execution_order[6], loss);
}

test "execution order gradient" {
    const constant = @import("constant.zig").constant;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, 5);
    const b = try mean(&graph, a);
    const c = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(session, &[_]Tensor{c[0]}, environment);
    std.testing.expectEqual(execution_order.len, 3);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c[0]);
}

test "execution order variable" {
    const constant = @import("constant.zig").constant;
    const variable = @import("variable.zig").variable;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, 5);
    const b = try variable(&graph, a);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(session, &[_]Tensor{b}, environment);
    std.testing.expectEqual(execution_order.len, 2);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
}

test "execution order assign" {
    const constant = @import("constant.zig").constant;
    const variable = @import("variable.zig").variable;
    const assign = @import("assign.zig").assign;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, 5);
    const b = try constant(f64, &graph, 10);
    const c = try variable(&graph, a);
    const d = try assign(&graph, c, b);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(session, &[_]Tensor{ d, c }, environment);
    std.testing.expectEqual(execution_order.len, 4);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], c);
    std.testing.expectEqual(execution_order[2], b);
    std.testing.expectEqual(execution_order[3], d);
}

test "execution order repeated tensors" {
    const constant = @import("constant.zig").constant;
    const add = @import("add.zig").add;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, 3);
    const b = try constant(f64, &graph, 5);
    const c = try add(&graph, a, b);
    const d = try add(&graph, c, c);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(session, &[_]Tensor{d}, environment);
    std.testing.expectEqual(execution_order.len, 4);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c);
    std.testing.expectEqual(execution_order[3], d);
}

test "execution order placeholder" {
    const constant = @import("constant.zig").constant;
    const placeholder = @import("placeholder.zig").placeholder;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, 3);
    const b = try constant(f64, &graph, 5);
    const c = try placeholder(&graph, &[_]usize{}, .f64);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();

    var environment = Environment.init(&session.arena.allocator);
    try environment.putNoClobber(c, a);
    const execution_order = try executionOrder(session, &[_]Tensor{c}, environment);
    std.testing.expectEqual(execution_order.len, 2);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], c);

    var environment2 = Environment.init(&session.arena.allocator);
    try environment2.putNoClobber(c, b);
    const execution_order2 = try executionOrder(session, &[_]Tensor{c}, environment2);
    std.testing.expectEqual(execution_order2.len, 2);
    std.testing.expectEqual(execution_order2[0], b);
    std.testing.expectEqual(execution_order2[1], c);
}

fn indexOf(needle: Tensor, haystack: []const Tensor) !usize {
    var i = haystack.len - 1;
    while (i > 0) : (i -= 1) if (std.meta.eql(haystack[i], needle))
        return i;
    return error.NotFound;
}

fn runConstant(session: Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    const constant = session.graph.constants.at(index);
    try cache.putNoClobber(current_tensor, constant);
}

fn runOperation(session: Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    const operation = session.graph.operations.at(index);
    const inputs = operation.inputs(operation);
    var values = try session.arena.child_allocator.alloc(CpuTensorUnion, inputs.len);
    defer session.arena.child_allocator.free(values);
    for (inputs) |input, i| values[i] = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, input);
    const result = try operation.forward(.{
        .operation = operation,
        .allocator = &session.arena.allocator,
        .values = values,
    });
    try cache.putNoClobber(current_tensor, result);
}

const GradientContext = struct {
    session: Session,
    cache: *Cache,
    gradient_caches: *GradientCaches,
    execution_order: []const Tensor,
    gradient_handle: GradientHandle,
    current_tensor: Tensor,
};

pub fn accumulateGradients(allocator: *Allocator, stored: CpuTensorUnion, incoming: CpuTensorUnion) !CpuTensorUnion {
    return switch (stored) {
        .f64 => |tensor| CpuTensorUnion.init(try eager.add(f64, allocator, tensor, incoming.f64)),
        .f32 => |tensor| CpuTensorUnion.init(try eager.add(f32, allocator, tensor, incoming.f32)),
        .f16 => |tensor| CpuTensorUnion.init(try eager.add(f16, allocator, tensor, incoming.f16)),
        else => return error.CannotDifferentiateIntegral,
    };
}

fn runGradient(context: GradientContext) !void {
    if (context.gradient_caches.getValue(context.gradient_handle.gradient)) |gradient_cache| {
        const gradient_operation = context.session.graph.gradients.at(context.gradient_handle.gradient);
        const value = try getValue(Cache, Tensor, CpuTensorUnion, gradient_cache, gradient_operation.with_respect_to[context.gradient_handle.index]);
        try context.cache.putNoClobber(context.current_tensor, value);
    } else {
        const allocator = &context.session.arena.allocator;
        var gradient_cache = Cache.init(allocator);
        errdefer gradient_cache.deinit();
        const gradient_operation = context.session.graph.gradients.at(context.gradient_handle.gradient);
        const of = gradient_operation.of;
        const one = switch (try getValue(Cache, Tensor, CpuTensorUnion, context.cache.*, of)) {
            .f64 => CpuTensorUnion.init(try eager.constant(f64, allocator, 1)),
            .f32 => CpuTensorUnion.init(try eager.constant(f64, allocator, 1)),
            .f16 => CpuTensorUnion.init(try eager.constant(f64, allocator, 1)),
            else => return error.CannotDifferentiateIntegral,
        };
        try gradient_cache.putNoClobber(of, one);
        var i = try indexOf(of, context.execution_order);
        while (i > 0) : (i -= 1) {
            const current = context.execution_order[i];
            switch (current.tensorType) {
                .operation => |index| {
                    const operation = context.session.graph.operations.at(index);
                    const inputs = operation.inputs(operation);
                    var forward_inputs = try allocator.alloc(CpuTensorUnion, inputs.len);
                    defer allocator.free(forward_inputs);
                    for (inputs) |input, j| forward_inputs[j] = try getValue(Cache, Tensor, CpuTensorUnion, context.cache.*, input);
                    const gradient_input = try getValue(Cache, Tensor, CpuTensorUnion, gradient_cache, current);
                    if (operation.backward) |backward| {
                        const gradients = try backward(.{
                            .operation = operation,
                            .allocator = allocator,
                            .gradient_input = gradient_input,
                            .forward_inputs = forward_inputs,
                        });
                        for (inputs) |input, j| {
                            const result = try gradient_cache.getOrPut(input);
                            if (!result.found_existing) {
                                result.kv.value = gradients[j];
                            } else {
                                result.kv.value = try accumulateGradients(allocator, result.kv.value, gradients[j]);
                            }
                        }
                    }
                },
                else => {},
            }
        }
        const value = try getValue(Cache, Tensor, CpuTensorUnion, gradient_cache, gradient_operation.with_respect_to[context.gradient_handle.index]);
        try context.cache.putNoClobber(context.current_tensor, value);
        try context.gradient_caches.putNoClobber(context.gradient_handle.gradient, gradient_cache);
    }
}

fn runVariable(session: Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    if (session.variableCache.getValue(current_tensor)) |current_value| {
        try cache.putNoClobber(current_tensor, current_value);
    } else {
        const variable = session.graph.variables.at(index);
        const current_value = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, variable.current_value);
        try cache.putNoClobber(current_tensor, current_value);
    }
}

fn runAssign(session: *Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    const assign = session.graph.assigns.at(index);
    const value = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, assign.value);
    try cache.putNoClobber(current_tensor, value);
    _ = try cache.put(assign.variable, value);
    _ = try session.variableCache.put(assign.variable, value);
}

fn runPlaceholder(session: Session, cache: *Cache, environment: Environment, current_tensor: Tensor) !void {
    const desired_tensor = try getValue(Environment, Tensor, Tensor, environment, current_tensor);
    const desired_value = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, desired_tensor);
    try cache.putNoClobber(current_tensor, desired_value);
}

pub const Session = struct {
    arena: *std.heap.ArenaAllocator,
    graph: *const Graph,
    variableCache: Cache,

    pub fn init(allocator: *std.mem.Allocator, graph: *const Graph) !Session {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Session{
            .arena = arena,
            .graph = graph,
            .variableCache = Cache.init(&arena.allocator),
        };
    }

    pub fn deinit(self: *Session) void {
        const child_allocator = self.arena.child_allocator;
        self.arena.deinit();
        child_allocator.destroy(self.arena);
    }

    const RunParameters = struct {
        environment: Environment = Environment.init(std.heap.page_allocator),
    };

    pub fn run(self: *Session, tensors: []const Tensor, parameters: RunParameters) ![]CpuTensorUnion {
        const allocator = &self.arena.allocator;
        const graph = self.graph;
        var cache = Cache.init(allocator);
        defer cache.deinit();
        var gradient_caches = GradientCaches.init(allocator);
        defer gradient_caches.deinit();
        const execution_order = try executionOrder(self.*, tensors, parameters.environment);
        for (execution_order) |current_tensor| {
            switch (current_tensor.tensorType) {
                .constant => |index| try runConstant(self.*, &cache, index, current_tensor),
                .operation => |index| try runOperation(self.*, &cache, index, current_tensor),
                .gradient_handle => |gradient_handle| try runGradient(.{
                    .session = self.*,
                    .cache = &cache,
                    .gradient_caches = &gradient_caches,
                    .execution_order = execution_order,
                    .gradient_handle = gradient_handle,
                    .current_tensor = current_tensor,
                }),
                .variable => |index| try runVariable(self.*, &cache, index, current_tensor),
                .assign => |index| try runAssign(self, &cache, index, current_tensor),
                .placeholder => try runPlaceholder(self.*, &cache, parameters.environment, current_tensor),
            }
        }
        const outputs = try self.arena.allocator.alloc(CpuTensorUnion, tensors.len);
        errdefer self.arena.allocator.free(outputs);
        for (tensors) |tensor, index| outputs[index] = try getValue(Cache, Tensor, CpuTensorUnion, cache, tensor);
        return outputs;
    }
};

test "session run" {
    const constant = @import("constant.zig").constant;
    const add = @import("add.zig").add;
    const matrixMultiply = @import("matrix_multiply.zig").matrixMultiply;
    const subtract = @import("subtract.zig").subtract;
    const absolute = @import("absolute.zig").absolute;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try constant(f64, &graph, .{
        .{ 0, 7, 3 },
        .{ 4, 5, 6 },
        .{ -10, -2, 0 },
    });
    const x = try constant(f64, &graph, .{
        .{1},
        .{2},
        .{3},
    });
    const h = try matrixMultiply(&graph, m, x);
    const b = try constant(f64, &graph, .{
        .{3},
        .{7},
        .{5},
    });
    const y_hat = try add(&graph, h, b);
    const y = try constant(f64, &graph, .{
        .{1},
        .{4},
        .{9},
    });
    const delta = try subtract(&graph, y, y_hat);
    const magnitude = try absolute(&graph, delta);
    const loss = try mean(&graph, magnitude);
    const gradients = try gradient(&graph, loss, &[_]Tensor{ m, b });
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{ loss, gradients[0], gradients[1] }, .{});
    const expected_loss = try eager.constant(f64, &arena.allocator, 26);
    const expected_m_gradient = try eager.constant(f64, &arena.allocator, .{
        .{ 1 / 3., 2 / 3., 1 },
        .{ 1 / 3., 2 / 3., 1 },
        .{ -1 / 3., -2 / 3., -1 },
    });
    const expected_b_gradient = try eager.constant(f64, &arena.allocator, .{
        .{1 / 3.},
        .{1 / 3.},
        .{-1 / 3.},
    });
    expectEqual(f64, actual[0].f64, expected_loss);
    expectEqual(f64, actual[1].f64, expected_m_gradient);
    expectEqual(f64, actual[2].f64, expected_b_gradient);
}

test "variable" {
    const constant = @import("constant.zig").constant;
    const variable = @import("variable.zig").variable;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try variable(&graph, a);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(&[_]Tensor{b}, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    expectEqual(f64, actual[0].f64, expected);
}

test "duplicate" {
    const constant = @import("constant.zig").constant;
    const add = @import("add.zig").add;
    const multiply = @import("multiply.zig").multiply;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try constant(f64, &graph, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const c = try constant(f64, &graph, .{
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const d = try multiply(&graph, a, b);
    const e = try multiply(&graph, a, c);
    const f = try add(&graph, d, e);
    const g = try mean(&graph, f);
    const gradients = try gradient(&graph, g, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(gradients, .{});
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 3.5, 4 },
        .{ 4.5, 5 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
