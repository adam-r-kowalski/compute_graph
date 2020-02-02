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

    const actual1 = try session.run(&[_]Tensor{ e, b });
    const expected1 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2, 3 },
        .{ 4, 5 },
    });
    expectEqual(f64, actual1[0].f64, expected1);
    expectEqual(f64, actual1[1].f64, expected1);

    const actual2 = try session.run(&[_]Tensor{ e, b });
    const expected2 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
    expectEqual(f64, actual2[1].f64, expected2);

    const actual3 = try session.run(&[_]Tensor{ e, b });
    const expected3 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 4, 5 },
        .{ 6, 7 },
    });
    expectEqual(f64, actual3[0].f64, expected3);
    expectEqual(f64, actual3[1].f64, expected3);
}

test "linear regression" {
    const add = @import("add.zig").add;
    const absolute = @import("absolute.zig").absolute;
    const multiply = @import("multiply.zig").multiply;
    const subtract = @import("subtract.zig").subtract;
    const constant = @import("constant.zig").constant;
    const variable = @import("variable.zig").variable;
    const gradient = @import("gradient.zig").gradient;
    const eager = @import("../eager.zig");
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try variable(&graph, try constant(&graph, @as(f64, 8)));
    const b = try variable(&graph, try constant(&graph, @as(f64, 6)));
    const x = try constant(&graph, @as(f64, 2));
    const y = try constant(&graph, @as(f64, 4));
    const y_hat = try add(&graph, try multiply(&graph, m, x), b);
    const loss = try absolute(&graph, try subtract(&graph, y, y_hat));
    const gradients = try gradient(&graph, loss, &[_]Tensor{ m, b });
    const step_size = try constant(&graph, @as(f64, 0.3));
    const dm = try multiply(&graph, gradients[0], step_size);
    const db = try multiply(&graph, gradients[1], step_size);
    const improve_m = try assign(&graph, m, try subtract(&graph, m, dm));
    const improve_b = try assign(&graph, b, try subtract(&graph, b, db));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();

    const actual = try session.run(&[_]Tensor{loss});
    const actual_loss = actual[0];
    expectEqual(f64, actual_loss.f64, try eager.constant(&arena.allocator, @as(f64, 1.8e+01)));

    var i: usize = 0;
    while (i < 100) : (i += 1)
        _ = try session.run(&[_]Tensor{ improve_m, improve_b });

    const actual1 = try session.run(&[_]Tensor{ loss, m, b });
    const actual_loss1 = actual1[0];
    const actual_m1 = actual1[1];
    const actual_b1 = actual1[2];
    expectEqual(f64, actual_loss1.f64, try eager.constant(&arena.allocator, @as(f64, 5.329070518200751e-15)));
    expectEqual(f64, actual_m1.f64, try eager.constant(&arena.allocator, @as(f64, 8.000000000000017e-01)));
    expectEqual(f64, actual_b1.f64, try eager.constant(&arena.allocator, @as(f64, 2.400000000000002)));
}
