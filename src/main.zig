const std = @import("std");

pub fn Tensor(comptime T: type) type {
    return union(enum) {
        constant: u64,
        operation: u64,
    };
}

pub fn Graph(comptime T: type) type {
    return struct {
        parent_allocator: *std.mem.Allocator,
        arena: *std.heap.ArenaAllocator,
        constants: std.ArrayList(Constant(T)),
        operations: std.ArrayList(*const Operation(T)),

        pub const elementType: type = T;

        pub fn init(allocator: *std.mem.Allocator) !Graph(T) {
            const arena = try allocator.create(std.heap.ArenaAllocator);
            arena.* = std.heap.ArenaAllocator.init(allocator);
            return Graph(T){
                .parent_allocator = allocator,
                .arena = arena,
                .constants = std.ArrayList(Constant(T)).init(&arena.allocator),
                .operations = std.ArrayList(*const Operation(T)).init(&arena.allocator),
            };
        }

        pub fn deinit(self: *Graph(T)) void {
            self.arena.deinit();
            self.parent_allocator.destroy(self.arena);
        }
    };
}

pub fn Constant(comptime T: type) type {
    return struct {
        value: T,
    };
}

pub fn constant(graph: var, value: var) !Tensor(@typeOf(graph.*).elementType) {
    const T = @typeOf(graph.*).elementType;
    try graph.constants.append(.{ .value = value });
    return Tensor(T){ .constant = graph.constants.count() - 1 };
}

test "constant" {
    var graph = try Graph(f64).init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);

    std.testing.expectEqual(graph.constants.at(x.constant).value, 5);
    std.testing.expectEqual(graph.constants.at(y.constant).value, 10);
}

pub fn Operation(comptime T: type) type {
    return struct {
        inputs: fn (self: *const Operation(T)) []const Tensor(T),
    };
}

pub fn Add(comptime T: type) type {
    return struct {
        operation: Operation(T),
        left: Tensor(T),
        right: Tensor(T),

        pub fn init(left: Tensor(T), right: Tensor(T)) Add(T) {
            return .{
                .operation = .{ .inputs = inputs },
                .left = left,
                .right = right,
            };
        }

        pub fn inputs(operation: *const Operation(T)) []const Tensor(T) {
            const self = @fieldParentPtr(Add(T), "operation", operation);
            return &[_]Tensor(T){ self.left, self.right };
        }
    };
}

pub fn add(graph: var, x: var, y: @typeOf(x)) !@typeOf(x) {
    const T = @typeOf(graph.*).elementType;
    var a = try graph.arena.allocator.create(Add(T));
    a.* = Add(T).init(x, y);
    try graph.operations.append(&a.operation);
    return Tensor(T){ .operation = graph.operations.count() - 1 };
}

test "add" {
    var graph = try Graph(f64).init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try add(&graph, x, y);

    const operation = graph.operations.at(z.operation);
    const a = @fieldParentPtr(Add(f64), "operation", operation);
    const left = graph.constants.at(a.left.constant);
    const right = graph.constants.at(a.right.constant);
    std.testing.expectEqual(graph.constants.at(x.constant), left);
    std.testing.expectEqual(graph.constants.at(y.constant), right);
}

pub fn Multiply(comptime T: type) type {
    return struct {
        operation: Operation(T),
        left: Tensor(T),
        right: Tensor(T),

        pub fn init(left: Tensor(T), right: Tensor(T)) Multiply(T) {
            return .{
                .operation = .{ .inputs = inputs },
                .left = left,
                .right = right,
            };
        }

        pub fn inputs(operation: *const Operation(T)) []const Tensor(T) {
            const self = @fieldParentPtr(Multiply(T), "operation", operation);
            return &[_]Tensor(T){ self.left, self.right };
        }
    };
}

pub fn multiply(graph: var, x: var, y: @typeOf(x)) !@typeOf(x) {
    const T = @typeOf(graph.*).elementType;
    var a = try graph.arena.allocator.create(Multiply(T));
    a.* = Multiply(T).init(x, y);
    try graph.operations.append(&a.operation);
    return Tensor(T){ .operation = graph.operations.count() - 1 };
}

test "multiply" {
    var graph = try Graph(f64).init(std.heap.page_allocator);
    defer graph.deinit();
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try multiply(&graph, x, y);

    const operation = graph.operations.at(z.operation);
    const a = @fieldParentPtr(Multiply(f64), "operation", operation);
    const left = graph.constants.at(a.left.constant);
    const right = graph.constants.at(a.right.constant);
    std.testing.expectEqual(graph.constants.at(x.constant), left);
    std.testing.expectEqual(graph.constants.at(y.constant), right);
}

pub fn main() !void {}
