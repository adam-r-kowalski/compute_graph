const std = @import("std");
const Operation = @import("operation.zig").Operation;
const CpuTensorUnion = @import("../eager.zig").CpuTensorUnion;
const Gradient = @import("gradient.zig").Gradient;

pub const Graph = struct {
    arena: *std.heap.ArenaAllocator,
    constants: std.ArrayList(CpuTensorUnion),
    operations: std.ArrayList(*const Operation),
    gradients: std.ArrayList(Gradient),

    pub fn init(allocator: *std.mem.Allocator) !Graph {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Graph{
            .arena = arena,
            .constants = std.ArrayList(CpuTensorUnion).init(&arena.allocator),
            .operations = std.ArrayList(*const Operation).init(&arena.allocator),
            .gradients = std.ArrayList(Gradient).init(&arena.allocator),
        };
    }

    pub fn deinit(self: *Graph) void {
        const child_allocator = self.arena.child_allocator;
        self.arena.deinit();
        child_allocator.destroy(self.arena);
    }
};
