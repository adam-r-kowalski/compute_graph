const std = @import("std");
const Operation = @import("operation.zig").Operation;

pub const Constant = struct {
    value: f64,
};

pub const Graph = struct {
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
