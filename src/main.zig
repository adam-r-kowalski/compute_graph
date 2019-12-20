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
};

const Add = struct {
    operation: Operation,
    nodes: [2]Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Add {
        return .{
            .operation = .{ .inputs = inputs },
            .nodes = .{ left.node, right.node },
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Add, "operation", operation);
        return &self.nodes;
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
            .operation = .{ .inputs = inputs },
            .nodes = .{ left.node, right.node },
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Multiply, "operation", operation);
        return &self.nodes;
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
            .operation = .{ .inputs = inputs },
            .nodes = .{ left.node, right.node },
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Subtract, "operation", operation);
        return &self.nodes;
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

const Session = struct {
    arena: *std.heap.ArenaAllocator,

    pub fn init(allocator: *std.mem.Allocator) !Session {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Session{
            .arena = arena,
        };
    }

    pub fn deinit(self: *Session) void {
        const child_allocator = self.arena.child_allocator;
        self.arena.deinit();
        child_allocator.destroy(self.arena);
    }
};

const topological_sort = struct {
    const Nodes = std.ArrayList(Node);
    const Set = std.AutoHashMap(Node, void);
    const Error = error{OutOfMemory};

    fn recurse(nodes: *Nodes, visited: *Set, graph: Graph, node: Node) Error!void {
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

    fn execution_order(session: Session, graph: Graph, tensor: var) ![]const Node {
        var nodes = Nodes.init(&session.arena.allocator);
        var visited = Set.init(session.arena.child_allocator);
        defer visited.deinit();
        try recurse(&nodes, &visited, graph, tensor.node);
        return nodes.toSlice();
    }
};

test "topological sort" {
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
    var session = try Session.init(allocator);
    defer session.deinit();
    const execution_order = try topological_sort.execution_order(session, graph, loss);
    std.testing.expectEqual(execution_order[0], y.node);
    std.testing.expectEqual(execution_order[1], m.node);
    std.testing.expectEqual(execution_order[2], x.node);
    std.testing.expectEqual(execution_order[3], h.node);
    std.testing.expectEqual(execution_order[4], b.node);
    std.testing.expectEqual(execution_order[5], y_hat.node);
    std.testing.expectEqual(execution_order[6], loss.node);
}

test "topological sort two nodes" {
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, 3);
    const b = try constant(&graph, 5);
    const c = try add(&graph, a, b);
    const d = try add(&graph, a, c);
    var session = try Session.init(allocator);
    defer session.deinit();
    const execution_order = try topological_sort.execution_order(session, graph, d);
    std.testing.expectEqual(execution_order[0], a.node);
    std.testing.expectEqual(execution_order[1], b.node);
    std.testing.expectEqual(execution_order[2], c.node);
    std.testing.expectEqual(execution_order[3], d.node);
}

pub fn main() !void {}
