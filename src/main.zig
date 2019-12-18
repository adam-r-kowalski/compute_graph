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
    left: Node,
    right: Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Add {
        return .{
            .operation = .{ .inputs = inputs },
            .left = left.node,
            .right = right.node,
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Add, "operation", operation);
        return &[_]Node{ self.left, self.right };
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
    const a = @fieldParentPtr(Add, "operation", operation);
    const left = graph.constants.at(a.left.constant);
    const right = graph.constants.at(a.right.constant);
    std.testing.expectEqual(graph.constants.at(x.node.constant), left);
    std.testing.expectEqual(graph.constants.at(y.node.constant), right);
}

const Multiply = struct {
    operation: Operation,
    left: Node,
    right: Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Multiply {
        return .{
            .operation = .{ .inputs = inputs },
            .left = left.node,
            .right = right.node,
        };
    }

    pub fn inputs(operation: *const Operation) []const Node {
        const self = @fieldParentPtr(Multiply, "operation", operation);
        return &[_]Node{ self.left, self.right };
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
    const a = @fieldParentPtr(Multiply, "operation", operation);
    const left = graph.constants.at(a.left.constant);
    const right = graph.constants.at(a.right.constant);
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

fn topologicalSort(graph: Graph, node: Node) void {
    const operation = graph.operations.at(node.operation);
    const inputs = operation.inputs(operation);
    const left = inputs[0];
    const right = inputs[1];
    std.debug.warn("\n{}\n{}\n{}\n", .{ left, right, node });
}

fn topologicalSort2(graph: Graph, node: Node) void {
    const operation = graph.operations.at(node.operation);
    const inputs = operation.inputs(operation);
    const nodes = .{ inputs[0], inputs[1], node };
    std.debug.warn("\n{}\n{}\n{}\n", .{ nodes[0], nodes[1], nodes[2] });
}

fn topologicalSort3(allocator: *std.mem.Allocator, graph: Graph, node: Node) !void {
    const operation = graph.operations.at(node.operation);
    const inputs = operation.inputs(operation);
    var nodes = std.ArrayList(Node).init(allocator);
    defer nodes.deinit();
    try nodes.append(inputs[0]);
    try nodes.append(inputs[1]);
    try nodes.append(node);
    std.debug.warn("\n{}\n{}\n{}\n", .{ nodes.at(0), nodes.at(1), nodes.at(2) });
}

fn topologicalSort4(allocator: *std.mem.Allocator, graph: Graph, node: Node) !void {
    const operation = graph.operations.at(node.operation);
    const inputs = operation.inputs(operation);
    var nodes = std.ArrayList(Node).init(allocator);
    defer nodes.deinit();
    const input_0 = inputs[0];
    const input_1 = inputs[1];
    try nodes.append(input_0);
    try nodes.append(input_1);
    try nodes.append(node);
    std.debug.warn("\n{}\n{}\n{}\n", .{ nodes.at(0), nodes.at(1), nodes.at(2) });
}

test "topologicalSort" {
    const allocator = std.heap.page_allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try add(&graph, x, y);
    var session = try Session.init(allocator);
    defer session.deinit();
    topologicalSort(graph, z.node);
    topologicalSort2(graph, z.node);
    try topologicalSort3(allocator, graph, z.node);
    try topologicalSort4(allocator, graph, z.node);
}

pub fn main() !void {}
