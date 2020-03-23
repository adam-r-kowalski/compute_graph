const std = @import("std");
const Allocator = std.mem.Allocator;
const gradient = @import("gradient.zig").gradient;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const GradientHandle = @import("tensor.zig").GradientHandle;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensorUnion = eager.CpuTensorUnion;
const constant = @import("constant.zig").constant;
const add = @import("add.zig").add;
const multiply = @import("multiply.zig").multiply;
const mean = @import("mean.zig").mean;
const matrixMultiply = @import("matrix_multiply.zig").matrixMultiply;
const subtract = @import("subtract.zig").subtract;
const absolute = @import("absolute.zig").absolute;
const variable = @import("variable.zig").variable;
const assign = @import("assign.zig").assign;
const placeholder = @import("placeholder.zig").placeholder;

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
                const variable_tensor = graph.variables.at(index);
                if (!visited.contains(variable_tensor.current_value))
                    try recurse(execution_order, visited, graph, environment, variable_tensor.current_value);
            },
            .assign => |index| {
                const assign_tensor = graph.assigns.at(index);
                if (!visited.contains(assign_tensor.variable))
                    try recurse(execution_order, visited, graph, environment, assign_tensor.variable);
                if (!visited.contains(assign_tensor.value))
                    try recurse(execution_order, visited, graph, environment, assign_tensor.value);
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

fn executionOrder(allocator: *Allocator, session: Session, tensors: []const Tensor, environment: Environment) ![]const Tensor {
    var execution_order = ExecutionOrder.Tensors.init(allocator);
    errdefer execution_order.deinit();
    var visited = ExecutionOrder.Visited.init(allocator);
    defer visited.deinit();
    for (tensors) |tensor|
        if (!visited.contains(tensor))
            try ExecutionOrder.recurse(&execution_order, &visited, session.graph, environment, tensor);
    return execution_order.toSlice();
}

test "execution order" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const m = try constant(f64, &graph, 3);
    const b = try constant(f64, &graph, 5);
    const y = try constant(f64, &graph, 35);
    const x = try constant(f64, &graph, 10);
    const h = try multiply(&graph, m, x);
    const y_hat = try add(&graph, h, b);
    const loss = try subtract(&graph, y, y_hat);
    var session = Session.init(&arena.allocator, &graph);
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(&arena.allocator, session, &[_]Tensor{loss}, environment);
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
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, 5);
    const b = try mean(&graph, a);
    const c = try gradient(&graph, b, &[_]Tensor{a});
    var session = Session.init(&arena.allocator, &graph);
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(&arena.allocator, session, &[_]Tensor{c[0]}, environment);
    std.testing.expectEqual(execution_order.len, 3);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c[0]);
}

test "execution order variable" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, 5);
    const b = try variable(&graph, a);
    var session = Session.init(&arena.allocator, &graph);
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(&arena.allocator, session, &[_]Tensor{b}, environment);
    std.testing.expectEqual(execution_order.len, 2);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
}

test "execution order assign" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, 5);
    const b = try constant(f64, &graph, 10);
    const c = try variable(&graph, a);
    const d = try assign(&graph, c, b);
    var session = Session.init(&arena.allocator, &graph);
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(&arena.allocator, session, &[_]Tensor{ d, c }, environment);
    std.testing.expectEqual(execution_order.len, 4);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], c);
    std.testing.expectEqual(execution_order[2], b);
    std.testing.expectEqual(execution_order[3], d);
}

test "execution order repeated tensors" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, 3);
    const b = try constant(f64, &graph, 5);
    const c = try add(&graph, a, b);
    const d = try add(&graph, c, c);
    var session = Session.init(&arena.allocator, &graph);
    const environment = Environment.init(&arena.allocator);
    const execution_order = try executionOrder(&arena.allocator, session, &[_]Tensor{d}, environment);
    std.testing.expectEqual(execution_order.len, 4);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c);
    std.testing.expectEqual(execution_order[3], d);
}

test "execution order placeholder" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, 3);
    const b = try constant(f64, &graph, 5);
    const c = try placeholder(&graph, &[_]usize{}, .f64);
    var session = Session.init(&arena.allocator, &graph);

    var environment = Environment.init(&arena.allocator);
    try environment.putNoClobber(c, a);
    const execution_order = try executionOrder(&arena.allocator, session, &[_]Tensor{c}, environment);
    std.testing.expectEqual(execution_order.len, 2);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], c);

    var environment2 = Environment.init(&arena.allocator);
    try environment2.putNoClobber(c, b);
    const execution_order2 = try executionOrder(&arena.allocator, session, &[_]Tensor{c}, environment2);
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
    const constant_tensor = session.graph.constants.at(index);
    try cache.putNoClobber(current_tensor, constant_tensor);
}

fn runOperation(allocator: *Allocator, session: Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    const operation = session.graph.operations.at(index);
    const inputs = operation.inputs(operation);
    var values = try allocator.alloc(CpuTensorUnion, inputs.len);
    defer allocator.free(values);
    for (inputs) |input, i| values[i] = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, input);
    const result = try operation.forward(.{
        .operation = operation,
        .allocator = allocator,
        .values = values,
    });
    try cache.putNoClobber(current_tensor, result);
}

const GradientContext = struct {
    allocator: *Allocator,
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
        const allocator = context.allocator;
        var gradient_cache = Cache.init(allocator);
        errdefer gradient_cache.deinit();
        const gradient_operation = context.session.graph.gradients.at(context.gradient_handle.gradient);
        const of = gradient_operation.of;
        const one = switch (try getValue(Cache, Tensor, CpuTensorUnion, context.cache.*, of)) {
            .f64 => CpuTensorUnion.init(try eager.constant(f64, allocator, 1)),
            .f32 => CpuTensorUnion.init(try eager.constant(f32, allocator, 1)),
            .f16 => CpuTensorUnion.init(try eager.constant(f16, allocator, 1)),
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
                    const forward_output = try getValue(Cache, Tensor, CpuTensorUnion, context.cache.*, current);
                    if (operation.backward) |backward| {
                        const gradients = try backward(.{
                            .operation = operation,
                            .allocator = allocator,
                            .gradient_input = gradient_input,
                            .forward_inputs = forward_inputs,
                            .forward_output = forward_output,
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
        const variable_tensor = session.graph.variables.at(index);
        const current_value = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, variable_tensor.current_value);
        try cache.putNoClobber(current_tensor, current_value);
    }
}

fn runAssign(session: *Session, cache: *Cache, index: usize, current_tensor: Tensor) !void {
    const assign_tensor = session.graph.assigns.at(index);
    const value = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, assign_tensor.value);
    try cache.putNoClobber(current_tensor, value);
    _ = try cache.put(assign_tensor.variable, value);
    _ = try session.variableCache.put(assign_tensor.variable, value);
}

fn runPlaceholder(session: Session, cache: *Cache, environment: Environment, current_tensor: Tensor) !void {
    const desired_tensor = try getValue(Environment, Tensor, Tensor, environment, current_tensor);
    const desired_value = try getValue(Cache, Tensor, CpuTensorUnion, cache.*, desired_tensor);
    try cache.putNoClobber(current_tensor, desired_value);
}

fn extractEnvironment(allocator: *Allocator, parameters: var) Environment {
    const T = @TypeOf(parameters);
    switch (@typeInfo(T)) {
        .Struct => {
            if (@hasField(T, "environment"))
                return parameters.environment;
        },
        else => {},
    }
    return Environment.init(allocator);
}

fn extractTensors(allocator: *Allocator, parameters: var) ![]const Tensor {
    const T = @TypeOf(parameters);
    switch (@typeInfo(T)) {
        .Struct => |s| {
            if (@hasField(T, "tensors"))
                return parameters.tensors;
            const tensors = try allocator.alloc(Tensor, s.fields.len);
            errdefer allocator.free(tensors);
            inline for (s.fields) |f, i|
                tensors[i] = @field(parameters, f.name);
            return tensors;
        },
        .Pointer => |p| {
            switch (@typeInfo(p.child)) {
                .Array, .Struct => return parameters,
                else => @compileError("session.run expected array of tensors"),
            }
        },
        else => @compileError("session.run expected tensor or array of tensors"),
    }
}

pub fn runTensor(allocator: *Allocator, session: *Session, tensor: Tensor, environment: Environment) !CpuTensorUnion {
    const graph = session.graph;
    var cache = Cache.init(allocator);
    defer cache.deinit();
    var gradient_caches = GradientCaches.init(allocator);
    defer gradient_caches.deinit();
    const execution_order = try executionOrder(allocator, session.*, &[_]Tensor{tensor}, environment);
    for (execution_order) |current_tensor| {
        switch (current_tensor.tensorType) {
            .constant => |index| try runConstant(session.*, &cache, index, current_tensor),
            .operation => |index| try runOperation(allocator, session.*, &cache, index, current_tensor),
            .gradient_handle => |gradient_handle| try runGradient(.{
                .allocator = allocator,
                .session = session.*,
                .cache = &cache,
                .gradient_caches = &gradient_caches,
                .execution_order = execution_order,
                .gradient_handle = gradient_handle,
                .current_tensor = current_tensor,
            }),
            .variable => |index| try runVariable(session.*, &cache, index, current_tensor),
            .assign => |index| try runAssign(session, &cache, index, current_tensor),
            .placeholder => try runPlaceholder(session.*, &cache, environment, current_tensor),
        }
    }
    return try getValue(Cache, Tensor, CpuTensorUnion, cache, tensor);
}

pub fn runTensors(allocator: *Allocator, session: *Session, tensors: []const Tensor, environment: Environment) ![]CpuTensorUnion {
    const graph = session.graph;
    var cache = Cache.init(allocator);
    defer cache.deinit();
    var gradient_caches = GradientCaches.init(allocator);
    defer gradient_caches.deinit();
    const execution_order = try executionOrder(allocator, session.*, tensors, environment);
    for (execution_order) |current_tensor| {
        switch (current_tensor.tensorType) {
            .constant => |index| try runConstant(session.*, &cache, index, current_tensor),
            .operation => |index| try runOperation(allocator, session.*, &cache, index, current_tensor),
            .gradient_handle => |gradient_handle| try runGradient(.{
                .allocator = allocator,
                .session = session.*,
                .cache = &cache,
                .gradient_caches = &gradient_caches,
                .execution_order = execution_order,
                .gradient_handle = gradient_handle,
                .current_tensor = current_tensor,
            }),
            .variable => |index| try runVariable(session.*, &cache, index, current_tensor),
            .assign => |index| try runAssign(session, &cache, index, current_tensor),
            .placeholder => try runPlaceholder(session.*, &cache, environment, current_tensor),
        }
    }
    const outputs = try allocator.alloc(CpuTensorUnion, tensors.len);
    errdefer allocator.free(outputs);
    for (tensors) |tensor, index| outputs[index] = try getValue(Cache, Tensor, CpuTensorUnion, cache, tensor);
    return outputs;
}

fn RunOutputType(comptime T: type) type {
    if (T == Tensor) return CpuTensorUnion;
    return []CpuTensorUnion;
}

pub const Session = struct {
    allocator: *Allocator,
    graph: *const Graph,
    variableCache: Cache,

    pub fn init(allocator: *std.mem.Allocator, graph: *const Graph) Session {
        return Session{
            .allocator = allocator,
            .graph = graph,
            .variableCache = Cache.init(allocator),
        };
    }

    pub fn deinit(self: *Session) void {
        self.variableCache.deinit();
    }

    pub fn run(self: *Session, parameters: var) !RunOutputType(@TypeOf(parameters)) {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        // defer arena.deinit();
        const environment = extractEnvironment(self.allocator, parameters);
        if (@TypeOf(parameters) == Tensor) {
            return try runTensor(&arena.allocator, self, parameters, environment);
        } else {
            const tensors = try extractTensors(self.allocator, parameters);
            return try runTensors(&arena.allocator, self, tensors, environment);
        }
    }
};

test "session run" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
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
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(.{ loss, gradients[0], gradients[1] });
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
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try variable(&graph, a);
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(b);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    expectEqual(f64, actual.f64, expected);
}

test "duplicate" {
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
    const c = try constant(f64, &graph, .{
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const d = try multiply(&graph, a, b);
    const e = try multiply(&graph, a, c);
    const f = try add(&graph, d, e);
    const g = try mean(&graph, f);
    const gradients = try gradient(&graph, g, &[_]Tensor{a});
    var session = Session.init(&arena.allocator, &graph);
    const actual = try session.run(gradients);
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 3.5, 4 },
        .{ 4.5, 5 },
    });
    expectEqual(f64, actual[0].f64, expected);
}
