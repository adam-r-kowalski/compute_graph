const std = @import("std");
const gradient = @import("gradient.zig").gradient;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const GradientHandle = @import("tensor.zig").GradientHandle;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensorUnion = eager.CpuTensorUnion;

const ExecutionOrder = struct {
    const Tensors = std.ArrayList(Tensor);
    const Visited = std.AutoHashMap(Tensor, void);
    const Error = error{OutOfMemory};

    fn recurse(tensors: *Tensors, visited: *Visited, graph: *const Graph, tensor: Tensor) Error!void {
        switch (tensor) {
            .operation => |index| {
                const operation = graph.operations.at(index);
                for (operation.inputs(operation)) |input|
                    if (!visited.contains(input))
                        try recurse(tensors, visited, graph, input);
            },
            .gradient_handle => |gradient_handle| {
                const of = graph.gradients.at(gradient_handle.gradient).of;
                if (!visited.contains(of))
                    try recurse(tensors, visited, graph, of);
            },
            .variable => |index| {
                const variable = graph.variables.at(index);
                if (!visited.contains(variable.current_value))
                    try recurse(tensors, visited, graph, variable.current_value);
            },
            .assign => |index| {
                const assign = graph.assigns.at(index);
                if (!visited.contains(assign.variable))
                    try recurse(tensors, visited, graph, assign.variable);
                if (!visited.contains(assign.value))
                    try recurse(tensors, visited, graph, assign.value);
            },
            else => {},
        }
        try visited.putNoClobber(tensor, undefined);
        try tensors.append(tensor);
    }
};

fn executionOrder(session: Session, tensors: []const Tensor) ![]const Tensor {
    var execution_order = ExecutionOrder.Tensors.init(&session.arena.allocator);
    errdefer execution_order.deinit();
    var visited = ExecutionOrder.Visited.init(session.arena.child_allocator);
    defer visited.deinit();
    for (tensors) |tensor|
        if (!visited.contains(tensor))
            try ExecutionOrder.recurse(&execution_order, &visited, session.graph, tensor);
    return execution_order.toSlice();
}

test "execution order" {
    const constant = @import("constant.zig").constant;
    const add = @import("add.zig").add;
    const multiply = @import("multiply.zig").multiply;
    const subtract = @import("subtract.zig").subtract;
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try constant(&graph, @as(f64, 3));
    const b = try constant(&graph, @as(f64, 5));
    const y = try constant(&graph, @as(f64, 35));
    const x = try constant(&graph, @as(f64, 10));
    const h = try multiply(&graph, m, x);
    const y_hat = try add(&graph, h, b);
    const loss = try subtract(&graph, y, y_hat);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const execution_order = try executionOrder(session, &[_]Tensor{loss});
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
    const a = try constant(&graph, @as(f64, 5));
    const b = try mean(&graph, a);
    const c = try gradient(&graph, b, &[_]Tensor{a});
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const execution_order = try executionOrder(session, &[_]Tensor{c[0]});
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
    const a = try constant(&graph, @as(f64, 5));
    const b = try variable(&graph, a);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const execution_order = try executionOrder(session, &[_]Tensor{b});
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
    const a = try constant(&graph, @as(f64, 5));
    const b = try constant(&graph, @as(f64, 10));
    const c = try variable(&graph, a);
    const d = try assign(&graph, c, b);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const execution_order = try executionOrder(session, &[_]Tensor{ d, c });
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
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, @as(f64, 3));
    const b = try constant(&graph, @as(f64, 5));
    const c = try add(&graph, a, b);
    const d = try add(&graph, c, c);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const execution_order = try executionOrder(session, &[_]Tensor{d});
    std.testing.expectEqual(execution_order.len, 4);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c);
    std.testing.expectEqual(execution_order[3], d);
}

const Cache = std.AutoHashMap(Tensor, CpuTensorUnion);
const GradientCaches = std.AutoHashMap(usize, Cache);

fn getValue(cache: Cache, key: Tensor) !CpuTensorUnion {
    if (cache.getValue(key)) |value| return value;
    return error.KeyNotFound;
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
    for (inputs) |input, i| values[i] = try getValue(cache.*, input);
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

fn runGradient(context: GradientContext) !void {
    if (context.gradient_caches.getValue(context.gradient_handle.gradient)) |gradient_cache| {
        const gradient_operation = context.session.graph.gradients.at(context.gradient_handle.gradient);
        const value = try getValue(gradient_cache, gradient_operation.with_respect_to[context.gradient_handle.index]);
        try context.cache.putNoClobber(context.current_tensor, value);
    } else {
        const allocator = &context.session.arena.allocator;
        var gradient_cache = Cache.init(allocator);
        errdefer gradient_cache.deinit();
        const gradient_operation = context.session.graph.gradients.at(context.gradient_handle.gradient);
        const of = gradient_operation.of;
        const one = switch (try getValue(context.cache.*, of)) {
            .f64 => CpuTensorUnion.init(try eager.constant(allocator, @as(f64, 1))),
            .f32 => CpuTensorUnion.init(try eager.constant(allocator, @as(f32, 1))),
            .f16 => CpuTensorUnion.init(try eager.constant(allocator, @as(f16, 1))),
            else => return error.CannotDifferentiateIntegral,
        };
        try gradient_cache.putNoClobber(of, one);
        var i = try indexOf(of, context.execution_order);
        while (i > 0) : (i -= 1) {
            const current = context.execution_order[i];
            switch (current) {
                .operation => |index| {
                    const operation = context.session.graph.operations.at(index);
                    const inputs = operation.inputs(operation);
                    var forward_inputs = try allocator.alloc(CpuTensorUnion, inputs.len);
                    defer allocator.free(forward_inputs);
                    for (inputs) |input, j| forward_inputs[j] = try getValue(context.cache.*, input);
                    const gradient_input = try getValue(gradient_cache, current);
                    if (operation.backward) |backward| {
                        const gradients = try backward(.{
                            .operation = operation,
                            .allocator = allocator,
                            .gradient_input = gradient_input,
                            .forward_inputs = forward_inputs,
                        });
                        for (inputs) |input, j| try gradient_cache.putNoClobber(input, gradients[j]);
                    } else return error.BackwardNotImplemented;
                },
                else => {},
            }
        }
        const value = try getValue(gradient_cache, gradient_operation.with_respect_to[context.gradient_handle.index]);
        try context.cache.putNoClobber(context.current_tensor, value);
        try context.gradient_caches.putNoClobber(context.gradient_handle.gradient, gradient_cache);
    }
}

fn runVariable(session: Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    if (session.variableCache.getValue(current_tensor)) |current_value| {
        try cache.putNoClobber(current_tensor, current_value);
    } else {
        const variable = session.graph.variables.at(index);
        const current_value = try getValue(cache.*, variable.current_value);
        try cache.putNoClobber(current_tensor, current_value);
    }
}

fn runAssign(session: *Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    const assign = session.graph.assigns.at(index);
    const value = try getValue(cache.*, assign.value);
    try cache.putNoClobber(current_tensor, value);
    _ = try cache.put(assign.variable, value);
    _ = try session.variableCache.put(assign.variable, value);
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

    const Context = struct {
        const Environment = std.AutoHashMap(Tensor, Tensor);

        tensors: []const Tensor,
        environment: Environment = Environment.init(std.heap.page_allocator),
    };

    pub fn run(self: *Session, context: Context) ![]CpuTensorUnion {
        const allocator = &self.arena.allocator;
        const graph = self.graph;
        var cache = Cache.init(allocator);
        defer cache.deinit();
        var gradient_caches = GradientCaches.init(allocator);
        defer gradient_caches.deinit();
        const execution_order = try executionOrder(self.*, context.tensors);
        for (execution_order) |current_tensor| {
            switch (current_tensor) {
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
            }
        }
        const outputs = try self.arena.allocator.alloc(CpuTensorUnion, context.tensors.len);
        errdefer self.arena.allocator.free(outputs);
        for (context.tensors) |tensor, index| outputs[index] = try getValue(cache, tensor);
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
    const m = try constant(&graph, [_][3]f64{
        .{ 0, 7, 3 },
        .{ 4, 5, 6 },
        .{ -10, -2, 0 },
    });
    const x = try constant(&graph, [_][1]f64{
        .{1},
        .{2},
        .{3},
    });
    const h = try matrixMultiply(&graph, m, x);
    const b = try constant(&graph, [_][1]f64{
        .{3},
        .{7},
        .{5},
    });
    const y_hat = try add(&graph, h, b);
    const y = try constant(&graph, [_][1]f64{
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
    const actual = try session.run(.{ .tensors = &[_]Tensor{ loss, gradients[0], gradients[1] } });
    const expected_loss = try eager.constant(&arena.allocator, @as(f64, 26));
    const expected_m_gradient = try eager.constant(&arena.allocator, [_][3]f64{
        .{ 1 / 3., 2 / 3., 1 },
        .{ 1 / 3., 2 / 3., 1 },
        .{ -1 / 3., -2 / 3., -1 },
    });
    const expected_b_gradient = try eager.constant(&arena.allocator, [_][1]f64{
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
    const a = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try variable(&graph, a);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(.{ .tensors = &[_]Tensor{b} });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
