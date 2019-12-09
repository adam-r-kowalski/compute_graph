const std = @import("std");

pub fn Constant(comptime T: type) type {
    return struct {
        value: T,
    };
}

pub fn Operation(comptime T: type) type {
    return struct {
        left: Tensor(T),
        right: Tensor(T),
    };
}

pub fn Tensor(comptime T: type) type {
    return union(enum) {
        constant: u64,
        operation: u64,
    };
}

pub fn Graph(comptime T: type) type {
    return struct {
        constants: std.ArrayList(Constant(T)),
        operations: std.ArrayList(Operation(T)),

        pub const elementType: type = T;

        pub fn init(allocator: *std.mem.Allocator) Graph(T) {
            return .{
                .constants = std.ArrayList(Constant(T)).init(allocator),
                .operations = std.ArrayList(Operation(T)).init(allocator),
            };
        }
    };
}

pub fn constant(graph: var, value: var) !Tensor(@typeOf(graph.*).elementType) {
    const T = @typeOf(graph.*).elementType;
    const c = try graph.constants.addOne();
    c.* = .{ .value = value };
    return Tensor(T){ .constant = graph.constants.count() - 1 };
}

test "constant" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    var graph = Graph(f64).init(allocator);
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    std.testing.expectEqual(graph.constants.at(x.constant).value, 5);
    std.testing.expectEqual(graph.constants.at(y.constant).value, 10);
}

pub fn add(graph: var, x: var, y: @typeOf(x)) !@typeOf(x) {
    const T = @typeOf(graph.*).elementType;
    const o = try graph.operations.addOne();
    o.* = .{ .left = x, .right = y };
    return Tensor(T){ .operation = graph.operations.count() - 1 };
}

test "add" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    var graph = Graph(f64).init(allocator);
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try add(&graph, x, y);

    const operation = graph.operations.at(z.operation);
    const left = graph.constants.at(operation.left.constant);
    const right = graph.constants.at(operation.right.constant);
    std.testing.expectEqual(graph.constants.at(x.constant), left);
    std.testing.expectEqual(graph.constants.at(y.constant), right);
}

pub fn main() !void {}
