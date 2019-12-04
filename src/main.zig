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

        pub fn init(allocator: *std.mem.Allocator) Graph(T) {
            return .{
                .constants = std.ArrayList(Constant(T)).init(allocator),
                .operations = std.ArrayList(Operation(T)).init(allocator),
            };
        }
    };
}

pub fn constant(comptime T: type, graph: *Graph(T), value: T) !Tensor(T) {
    try graph.constants.append(.{ .value = value });
    return Tensor(T){ .shape = &[_]u64{} };
}

pub fn add(comptime T: type, graph: *Graph(T), x: Tensor(T), y: Tensor(T)) !Tensor(T) {
    try graph.operations.append(.{ .left = x, .right = y });
    return Tensor(T){ .shape = &[_]u64{} };
}

test "create graph" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    const T = f64;
    var graph = Graph(T).init(allocator);
    const x = try constant(T, &graph, 5);
    const y = try constant(T, &graph, 10);
    const z = try add(T, &graph, x, y);
}

pub fn main() !void {}
