const std = @import("std");

pub fn Tensor(comptime T: type, comptime rank: u64) type {
    return Node;
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

const Constant = union(enum) {
    f64: f64,
};

pub fn constant(graph: var, value: var) !Tensor(f64, 0) {
    try graph.constants.append(.{ .f64 = value });
    return Tensor(f64, 0){ .constant = graph.constants.len - 1 };
}

test "constant" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);

    std.testing.expectEqual(graph.constants.at(x.constant).f64, 5);
    std.testing.expectEqual(graph.constants.at(y.constant).f64, 10);
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
            .left = left,
            .right = right,
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
    return Tensor(f64, 0){ .operation = graph.operations.len - 1 };
}

test "add" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try add(&graph, x, y);

    const operation = graph.operations.at(z.operation);
    const a = @fieldParentPtr(Add, "operation", operation);
    const left = graph.constants.at(a.left.constant);
    const right = graph.constants.at(a.right.constant);
    std.testing.expectEqual(graph.constants.at(x.constant), left);
    std.testing.expectEqual(graph.constants.at(y.constant), right);
}

const Multiply = struct {
    operation: Operation,
    left: Node,
    right: Node,

    pub fn init(left: Tensor(f64, 0), right: Tensor(f64, 0)) Multiply {
        return .{
            .operation = .{ .inputs = inputs },
            .left = left,
            .right = right,
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
    return Tensor(f64, 0){ .operation = graph.operations.len - 1 };
}

test "multiply" {
    var graph = try Graph.init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try multiply(&graph, x, y);

    const operation = graph.operations.at(z.operation);
    const a = @fieldParentPtr(Multiply, "operation", operation);
    const left = graph.constants.at(a.left.constant);
    const right = graph.constants.at(a.right.constant);
    std.testing.expectEqual(graph.constants.at(x.constant), left);
    std.testing.expectEqual(graph.constants.at(y.constant), right);
}

pub fn main() !void {}
