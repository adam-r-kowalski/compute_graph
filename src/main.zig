const std = @import("std");

pub fn Tensor(comptime S: type, comptime r: u64) type {
    return struct {
        node: Node,
    };
}

const Node = union(enum) {
    constant: u64,
    operation: u64,
};

const Graph = struct {
    arena: *std.heap.ArenaAllocator,
    constants: std.ArrayList(Constant),
    operations: std.ArrayList(*const Operation),

    pub fn init(allocator: *std.mem.Allocator) !Graph {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Graph{
            .arena = arena,
            .constants = std.ArrayList(Constant).init(&arena.allocator),
            .operations = std.ArrayList(*const Operation).init(&arena.allocator),
        };
    }

    pub fn deinit(self: *Graph) void {
        const child_allocator = self.arena.child_allocator;
        self.arena.deinit();
        child_allocator.destroy(self.arena);
    }
};

const Constant = struct {
    value: f64,
};

pub fn constant(graph: var, value: var) !Tensor(f64, 0) {
    try graph.constants.append(.{ .value = value });
    const node = Node{ .constant = graph.constants.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "constant" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    std.testing.expectEqual(graph.constants.at(x.node.constant).value, 5);
    std.testing.expectEqual(graph.constants.at(y.node.constant).value, 10);
}

const Operation = struct {
    inputs: fn (self: *const Operation) []const Node,
    forward: fn (self: *const Operation, values: []const f64) f64,
};

const Add = struct {
    operation: Operation,
    nodes: [2]Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Add {
        return .{
            .operation = .{
                .inputs = inputs,
                .forward = forward,
            },
            .nodes = .{ left.node, right.node },
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Add, "operation", operation);
        return &self.nodes;
    }

    pub fn forward(operation: *const Operation, values: []const f64) f64 {
        std.debug.assert(values.len == 2);
        return values[0] + values[1];
    }
};

pub fn add(graph: var, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    var a = try graph.arena.allocator.create(Add);
    a.* = Add.init(x, y);
    try graph.operations.append(&a.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "add" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try add(&graph, x, y);
    const operation = graph.operations.at(z.node.operation);
    const nodes = operation.inputs(operation);
    const left = graph.constants.at(nodes[0].constant);
    const right = graph.constants.at(nodes[1].constant);
    std.testing.expectEqual(graph.constants.at(x.node.constant), left);
    std.testing.expectEqual(graph.constants.at(y.node.constant), right);
}

const Multiply = struct {
    operation: Operation,
    nodes: [2]Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Multiply {
        return .{
            .operation = .{
                .inputs = inputs,
                .forward = forward,
            },
            .nodes = .{ left.node, right.node },
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Multiply, "operation", operation);
        return &self.nodes;
    }

    pub fn forward(operation: *const Operation, values: []const f64) f64 {
        std.debug.assert(values.len == 2);
        return values[0] * values[1];
    }
};

pub fn multiply(graph: var, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    var a = try graph.arena.allocator.create(Multiply);
    a.* = Multiply.init(x, y);
    try graph.operations.append(&a.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "multiply" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try multiply(&graph, x, y);
    const operation = graph.operations.at(z.node.operation);
    const nodes = operation.inputs(operation);
    const left = graph.constants.at(nodes[0].constant);
    const right = graph.constants.at(nodes[1].constant);
    std.testing.expectEqual(graph.constants.at(x.node.constant), left);
    std.testing.expectEqual(graph.constants.at(y.node.constant), right);
}

const Subtract = struct {
    operation: Operation,
    nodes: [2]Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Subtract {
        return .{
            .operation = .{
                .inputs = inputs,
                .forward = forward,
            },
            .nodes = .{ left.node, right.node },
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Subtract, "operation", operation);
        return &self.nodes;
    }

    pub fn forward(operation: *const Operation, values: []const f64) f64 {
        std.debug.assert(values.len == 2);
        return values[0] - values[1];
    }
};

pub fn subtract(graph: var, x: var, y: @TypeOf(x)) !@TypeOf(x) {
    var a = try graph.arena.allocator.create(Subtract);
    a.* = Subtract.init(x, y);
    try graph.operations.append(&a.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "subtract" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try subtract(&graph, x, y);
    const operation = graph.operations.at(z.node.operation);
    const nodes = operation.inputs(operation);
    const left = graph.constants.at(nodes[0].constant);
    const right = graph.constants.at(nodes[1].constant);
    std.testing.expectEqual(graph.constants.at(x.node.constant), left);
    std.testing.expectEqual(graph.constants.at(y.node.constant), right);
}

const Absolute = struct {
    operation: Operation,
    nodes: [1]Node,

    pub fn init(input: Tensor(f64, 0)) Absolute {
        return .{
            .operation = .{
                .inputs = inputs,
                .forward = forward,
            },
            .nodes = .{input.node},
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Absolute, "operation", operation);
        return &self.nodes;
    }

    pub fn forward(operation: *const Operation, values: []const f64) f64 {
        std.debug.assert(values.len == 1);
        return std.math.absFloat(values[0]);
    }
};

pub fn absolute(graph: var, input: var) !@TypeOf(input) {
    var a = try graph.arena.allocator.create(Absolute);
    a.* = Absolute.init(input);
    try graph.operations.append(&a.operation);
    const node = Node{ .operation = graph.operations.len - 1 };
    return Tensor(f64, 0){ .node = node };
}

test "absolute" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const a = try constant(&graph, -5);
    const b = try absolute(&graph, a);
    const operation = graph.operations.at(b.node.operation);
    const nodes = operation.inputs(operation);
    const input = graph.constants.at(nodes[0].constant);
    std.testing.expectEqual(graph.constants.at(a.node.constant), input);
}

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
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try constant(&graph, 3);
    const b = try constant(&graph, 5);
    const y = try constant(&graph, 35);
    const x = try constant(&graph, 10);
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
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, 3);
    const b = try constant(&graph, 5);
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

fn getValue(map: var, key: var) !f64 {
    if (map.getValue(key)) |value| return value;
    return error.KeyNotFound;
}

const Session = struct {
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

    pub fn run(self: Session, tensor: var) !f64 {
        const allocator = self.arena.child_allocator;
        const graph = self.graph;
        var cache = std.AutoHashMap(Node, f64).init(allocator);
        defer cache.deinit();
        for (try executionOrder(self, tensor)) |node| {
            switch (node) {
                .constant => |c| {
                    const value = graph.constants.at(c).value;
                    try cache.putNoClobber(node, value);
                },
                .operation => |o| {
                    const op = graph.operations.at(o);
                    var values = std.ArrayList(f64).init(allocator);
                    defer values.deinit();
                    for (op.inputs(op)) |input| {
                        const value = try getValue(cache, input);
                        try values.append(value);
                    }
                    try cache.putNoClobber(node, op.forward(op, values.toSlice()));
                },
            }
        }
        return try getValue(cache, tensor.node);
    }
};

test "session run" {
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try constant(&graph, 3);
    const b = try constant(&graph, 5);
    const y = try constant(&graph, 25);
    const x = try constant(&graph, 10);
    const h = try multiply(&graph, m, x);
    const y_hat = try add(&graph, h, b);
    const delta = try subtract(&graph, y, y_hat);
    const loss = try absolute(&graph, delta);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    const output = try session.run(loss);
    std.testing.expectEqual(output, 10);
}
