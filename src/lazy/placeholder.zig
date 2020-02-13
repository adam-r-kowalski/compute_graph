const std = @import("std");
// TODO(Adam): Clean up circular dependency
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Session = @import("session.zig").Session;
const Environment = @import("session.zig").Environment;
const expectEqual = @import("../testing.zig").expectEqual;

pub const Placeholder = struct {
    shape: []const usize,
};

pub fn placeholder(graph: *Graph, shape: []const usize) !Tensor {
    var placeholder_shape = try graph.arena.allocator.alloc(usize, shape.len);
    std.mem.copy(usize, placeholder_shape, shape);
    try graph.placeholders.append(Placeholder{
        .shape = shape,
    });
    return Tensor{
        .tensorType = .{ .placeholder = graph.placeholders.len - 1 },
        .shape = shape,
    };
}

test "placeholder" {
    const constant = @import("constant.zig").constant;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(&graph, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try constant(&graph, [_][2]f64{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const c = try placeholder(&graph, &[_]usize{ 2, 2 });
    std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 2 }));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();

    var environment = Environment.init(&session.arena.allocator);
    try environment.putNoClobber(c, a);
    const actual = try session.run(.{
        .tensors = &[_]Tensor{c},
        .environment = environment,
    });
    const expected = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    expectEqual(f64, actual[0].f64, expected);

    var environment2 = Environment.init(&session.arena.allocator);
    try environment2.putNoClobber(c, b);
    const actual2 = try session.run(.{
        .tensors = &[_]Tensor{c},
        .environment = environment2,
    });
    const expected2 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
}
