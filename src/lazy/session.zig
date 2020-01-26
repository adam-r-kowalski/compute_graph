const std = @import("std");
const gradient = @import("gradient.zig").gradient;
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensorUnion = eager.CpuTensorUnion;

const ExecutionOrder = struct {
    const Tensors = std.ArrayList(Tensor);
    const Visited = std.AutoHashMap(Tensor, void);
    const Error = error{OutOfMemory};

    fn recurse(tensors: *Tensors, visited: *Visited, graph: *const Graph, tensor: Tensor) Error!void {
        switch (tensor) {
            .operation => |o| {
                const operation = graph.operations.at(o);
                for (operation.inputs(operation)) |input|
                    if (!visited.contains(input))
                        try recurse(tensors, visited, graph, input);
            },
            .gradient_handle => |g| {
                const of = graph.gradients.at(g.gradient).of;
                if (!visited.contains(of))
                    try recurse(tensors, visited, graph, of);
            },
            else => {},
        }
        try visited.putNoClobber(tensor, undefined);
        try tensors.append(tensor);
    }
};

fn executionOrder(session: Session, tensor: Tensor) ![]const Tensor {
    var tensors = ExecutionOrder.Tensors.init(&session.arena.allocator);
    errdefer tensors.deinit();
    var visited = ExecutionOrder.Visited.init(session.arena.child_allocator);
    defer visited.deinit();
    try ExecutionOrder.recurse(&tensors, &visited, session.graph, tensor);
    return tensors.toSlice();
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
    const execution_order = try executionOrder(session, loss);
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
    const execution_order = try executionOrder(session, c[0]);
    std.testing.expectEqual(execution_order.len, 3);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c[0]);
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
    const execution_order = try executionOrder(session, d);
    std.testing.expectEqual(execution_order.len, 4);
    std.testing.expectEqual(execution_order[0], a);
    std.testing.expectEqual(execution_order[1], b);
    std.testing.expectEqual(execution_order[2], c);
    std.testing.expectEqual(execution_order[3], d);
}

fn getValue(map: var, key: var) !CpuTensorUnion {
    if (map.getValue(key)) |value| return value;
    return error.KeyNotFound;
}

pub const Session = struct {
    arena: *std.heap.ArenaAllocator,
    graph: *const Graph,

    pub fn init(allocator: *std.mem.Allocator, graph: *const Graph) !Session {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Session{
            .arena = arena,
            .graph = graph,
        };
    }

    pub fn deinit(self: *Session) void {
        const child_allocator = self.arena.child_allocator;
        self.arena.deinit();
        child_allocator.destroy(self.arena);
    }

    pub fn run(self: Session, tensor: Tensor) !CpuTensorUnion {
        const allocator = self.arena.child_allocator;
        const graph = self.graph;
        var cache = std.AutoHashMap(Tensor, CpuTensorUnion).init(allocator);
        defer cache.deinit();
        var i: usize = 0;
        const tensors = try executionOrder(self, tensor);
        while (i < tensors.len) {
            const current_tensor = tensors[i];
            switch (current_tensor) {
                .constant => |c| {
                    const value = graph.constants.at(c);
                    try cache.putNoClobber(current_tensor, value);
                },
                .operation => |o| {
                    const op = graph.operations.at(o);
                    var values = std.ArrayList(CpuTensorUnion).init(allocator);
                    defer values.deinit();
                    for (op.inputs(op)) |input| {
                        const value = try getValue(cache, input);
                        try values.append(value);
                    }
                    const result = try op.forward(.{
                        .op = op,
                        .allocator = &self.arena.allocator,
                        .values = values.toSlice(),
                    });
                    try cache.putNoClobber(current_tensor, result);
                },
                .gradient_handle => |g| {
                    var gradient_cache = std.AutoHashMap(Tensor, CpuTensorUnion).init(allocator);
                    defer gradient_cache.deinit();
                    const gradient_operation = graph.gradients.at(g.gradient);

                    const of = gradient_operation.of;
                    const one = try eager.constant(allocator, @as(f64, 1));
                    try gradient_cache.putNoClobber(of, CpuTensorUnion.init(one));

                    const op = graph.operations.at(of.operation);
                    var forward_inputs = std.ArrayList(CpuTensorUnion).init(allocator);
                    defer forward_inputs.deinit();
                    for (op.inputs(op)) |input| {
                        const value = try getValue(cache, input);
                        try forward_inputs.append(value);
                    }

                    const gradient_input = try getValue(gradient_cache, of);

                    if (op.backward) |backward| {
                        const gradients = try backward(.{
                            .op = op,
                            .allocator = &self.arena.allocator,
                            .gradient_input = gradient_input,
                            .forward_inputs = forward_inputs.toSlice(),
                        });
                        try gradient_cache.putNoClobber(gradient_operation.with_respect_to[0], gradients[0]);
                    }

                    const value = try getValue(gradient_cache, gradient_operation.with_respect_to[0]);
                    try cache.putNoClobber(current_tensor, value);
                },
            }
            i += 1;
        }
        return try getValue(cache, tensor);
    }
};

test "session run" {
    const constant = @import("constant.zig").constant;
    const add = @import("add.zig").add;
    const matrix_multiply = @import("matrix_multiply.zig").matrix_multiply;
    const subtract = @import("subtract.zig").subtract;
    const absolute = @import("absolute.zig").absolute;
    const mean = @import("mean.zig").mean;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try constant(&graph, [_][3]i64{
        .{ 0, 7, 3 },
        .{ 4, 5, 6 },
        .{ -10, -2, 0 },
    });
    const x = try constant(&graph, [_][1]i64{
        .{1},
        .{2},
        .{3},
    });
    const h = try matrix_multiply(&graph, m, x);
    const b = try constant(&graph, [_][1]i64{
        .{3},
        .{7},
        .{5},
    });
    const y_hat = try add(&graph, h, b);
    const y = try constant(&graph, [_][1]i64{
        .{1},
        .{4},
        .{9},
    });
    const delta = try subtract(&graph, y, y_hat);
    const magnitude = try absolute(&graph, delta);
    const loss = try mean(&graph, magnitude);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const actual = try session.run(loss);
    const expected = try eager.constant(&arena.allocator, @as(f64, 26));
    expectEqual(f64, actual.f64, expected);
}
