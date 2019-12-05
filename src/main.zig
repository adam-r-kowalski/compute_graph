const std = @import("std");

pub fn Tensor(comptime T: type) type {
    return struct {
        shape: []const u64,
    };
}

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
    try graph.constants.append(.{ .value = value });
    return Tensor(T){ .shape = &[_]u64{} };
}

pub fn add(graph: var, x: var, y: @typeOf(x)) !@typeOf(x) {
    try graph.operations.append(.{ .left = x, .right = y });
    return @typeOf(x){ .shape = &[_]u64{} };
}

test "create graph" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    var graph = Graph(f64).init(allocator);
    const x = try constant(&graph, 5);
    const y = try constant(&graph, 10);
    const z = try add(&graph, x, y);
}

pub fn main() !void {}
