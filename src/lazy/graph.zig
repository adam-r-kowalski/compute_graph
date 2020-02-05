const std = @import("std");
const Operation = @import("operation.zig").Operation;
const CpuTensorUnion = @import("../eager.zig").CpuTensorUnion;
// TODO(Adam): Clean up circular dependency
const Gradient = @import("gradient.zig").Gradient;
// TODO(Adam): Clean up circular dependency
const Variable = @import("variable.zig").Variable;
// TODO(Adam): Clean up circular dependency
const Assign = @import("assign.zig").Assign;
// TODO(Adam): Clean up circular dependency
const Placeholder = @import("placeholder.zig").Placeholder;

pub const Graph = struct {
    arena: *std.heap.ArenaAllocator,
    constants: std.ArrayList(CpuTensorUnion),
    operations: std.ArrayList(*const Operation),
    gradients: std.ArrayList(Gradient),
    variables: std.ArrayList(Variable),
    assigns: std.ArrayList(Assign),
    placeholders: std.ArrayList(Placeholder),

    pub fn init(allocator: *std.mem.Allocator) !Graph {
        const arena = try allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        return Graph{
            .arena = arena,
            .constants = std.ArrayList(CpuTensorUnion).init(&arena.allocator),
            .operations = std.ArrayList(*const Operation).init(&arena.allocator),
            .gradients = std.ArrayList(Gradient).init(&arena.allocator),
            .variables = std.ArrayList(Variable).init(&arena.allocator),
            .assigns = std.ArrayList(Assign).init(&arena.allocator),
            .placeholders = std.ArrayList(Placeholder).init(&arena.allocator),
        };
    }

    pub fn deinit(self: *Graph) void {
        const child_allocator = self.arena.child_allocator;
        self.arena.deinit();
        child_allocator.destroy(self.arena);
    }
};
