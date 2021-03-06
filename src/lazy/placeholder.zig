const std = @import("std");
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const ScalarType = tensor.ScalarType;
const Session = @import("session.zig").Session;
const Environment = @import("session.zig").Environment;
const expectEqual = @import("../testing.zig").expectEqual;
const constant = @import("constant.zig").constant;
const eager = @import("../eager.zig");

pub const Placeholder = struct {
    shape: []const usize,
};

pub fn placeholder(graph: *Graph, shape: []const usize, scalarType: ScalarType) !Tensor {
    var placeholder_shape = try graph.arena.allocator.alloc(usize, shape.len);
    std.mem.copy(usize, placeholder_shape, shape);
    try graph.placeholders.append(Placeholder{
        .shape = shape,
    });
    return Tensor{
        .tensorType = .{ .placeholder = graph.placeholders.len - 1 },
        .shape = shape,
        .scalarType = scalarType,
    };
}

test "placeholder" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try constant(f64, &graph, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const c = try placeholder(&graph, &[_]usize{ 2, 2 }, .f64);
    std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 2 }));
    var session = Session.init(&arena.allocator, &graph);

    var environment = Environment.init(&arena.allocator);
    try environment.putNoClobber(c, a);
    const actual = try session.run(.{
        .tensors = &[_]Tensor{c},
        .environment = environment,
    });
    const expected = try eager.constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    expectEqual(f64, actual[0].f64, expected);

    var environment2 = Environment.init(&arena.allocator);
    try environment2.putNoClobber(c, b);
    const actual2 = try session.run(.{
        .tensors = &[_]Tensor{c},
        .environment = environment2,
    });
    const expected2 = try eager.constant(f64, &arena.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
}
