const std = @import("std");
const Node = @import("node.zig").Node;
const gradient = @import("gradient.zig");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const eager = @import("../eager.zig");
const expectEqual = @import("../testing.zig").expectEqual;
const CpuTensorUnion = eager.CpuTensorUnion;

const ExecutionOrder = struct {
    const Nodes = std.ArrayList(Node);
    const Visited = std.AutoHashMap(Node, void);
    const Error = error{OutOfMemory};

    fn recurse(nodes: *Nodes, visited: *Visited, graph: *const Graph, node: Node) Error!void {
        switch (node) {
            .operation => |o| {
                const operation = graph.operations.at(o);
                for (operation.inputs(operation)) |input|
                    if (!visited.contains(input))
                        try recurse(nodes, visited, graph, input);
            },
            else => {},
        }
        try visited.putNoClobber(node, undefined);
        try nodes.append(node);
    }
};

fn executionOrder(session: Session, tensor: var) ![]const Node {
    var nodes = ExecutionOrder.Nodes.init(&session.arena.allocator);
    var visited = ExecutionOrder.Visited.init(session.arena.child_allocator);
    defer visited.deinit();
    try ExecutionOrder.recurse(&nodes, &visited, session.graph, tensor.node);
    return nodes.toSlice();
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
    std.testing.expectEqual(execution_order[0], y.node);
    std.testing.expectEqual(execution_order[1], m.node);
    std.testing.expectEqual(execution_order[2], x.node);
    std.testing.expectEqual(execution_order[3], h.node);
    std.testing.expectEqual(execution_order[4], b.node);
    std.testing.expectEqual(execution_order[5], y_hat.node);
    std.testing.expectEqual(execution_order[6], loss.node);
}

test "execution order repeated nodes" {
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
    std.testing.expectEqual(execution_order[0], a.node);
    std.testing.expectEqual(execution_order[1], b.node);
    std.testing.expectEqual(execution_order[2], c.node);
    std.testing.expectEqual(execution_order[3], d.node);
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

    pub fn run(self: Session, tensor: var) !CpuTensorUnion {
        const allocator = self.arena.child_allocator;
        const graph = self.graph;
        var cache = std.AutoHashMap(Node, CpuTensorUnion).init(allocator);
        defer cache.deinit();
        var i: usize = 0;
        const nodes = try executionOrder(self, tensor);
        while (i < nodes.len) {
            const node = nodes[i];
            switch (node) {
                .constant => |c| {
                    const value = graph.constants.at(c);
                    try cache.putNoClobber(node, value);
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
                    try cache.putNoClobber(node, result);
                },
                .gradient => |g| {
                    var gradient_cache = std.AutoHashMap(Node, CpuTensorUnion).init(allocator);
                    defer gradient_cache.deinit();
                    const gradient_operation = graph.gradients.at(g);
                    const op = graph.operations.at(gradient_operation.of.operation);

                    const value = gradient_operation.with_respect_to.constant;
                    const constant = graph.constants.at(value);

                    if (op.backward) |backward| {
                        _ = try backward(.{
                            .op = op,
                            .allocator = &self.arena.allocator,
                            .value = constant,
                        });
                    }

                    try cache.putNoClobber(node, constant);
                }
            }
            i += 1;
        }
        return try getValue(cache, tensor.node);
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
    expectEqual(actual.f64, expected);
}
