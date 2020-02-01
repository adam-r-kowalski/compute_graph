const std = @import("std");
// TODO(Adam): Clean up circular dependency
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Session = @import("session.zig").Session;
const expectEqual = @import("../testing.zig").expectEqual;

pub const Assign = struct {
    variable: Tensor,
    value: Tensor,
};

pub fn assign(graph: *Graph, variable: Tensor, value: Tensor) !Tensor {
    try graph.assigns.append(Assign{
        .variable = variable,
        .value = value,
    });
    return Tensor{ .assign = graph.assigns.len - 1 };
}

test "assign" {
    const add = @import("add.zig").add;
    const constant = @import("constant.zig").constant;
    const variable = @import("variable.zig").variable;
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
    const b = try variable(&graph, a);
    const c = try constant(&graph, [_][2]f64{
        .{ 1, 1 },
        .{ 1, 1 },
    });
    const d = try add(&graph, b, c);
    const e = try assign(&graph, b, d);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();
    // TODO(Adam):
    // Running assign op and variable should not error
    // try session.run(&[_]Tensor{e, b});
    const actual1 = try session.run(&[_]Tensor{e});
    const expected1 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2, 3 },
        .{ 4, 5 },
    });
    const actual2 = try session.run(&[_]Tensor{e});
    const expected2 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f64, actual1[0].f64, expected1);
    // expectEqual(f64, actual2[0].f64, expected2);
}
