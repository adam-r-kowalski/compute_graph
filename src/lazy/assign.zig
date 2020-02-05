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

    const actual1 = try session.run(.{ .tensors = &[_]Tensor{ e, b } });
    const expected1 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2, 3 },
        .{ 4, 5 },
    });
    expectEqual(f64, actual1[0].f64, expected1);
    expectEqual(f64, actual1[1].f64, expected1);

    const actual2 = try session.run(.{ .tensors = &[_]Tensor{ e, b } });
    const expected2 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
    expectEqual(f64, actual2[1].f64, expected2);

    const actual3 = try session.run(.{ .tensors = &[_]Tensor{ e, b } });
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
    const placeholder = @import("placeholder.zig").placeholder;
    const gradient = @import("gradient.zig").gradient;
    const eager = @import("../eager.zig");
    const Environment = @import("session.zig").Environment;
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try variable(&graph, try constant(&graph, @as(f64, 8)));
    const b = try variable(&graph, try constant(&graph, @as(f64, 6)));

    const x = try placeholder(&graph, &[_]usize{});
    const x1 = try constant(&graph, @as(f64, 1));
    const x2 = try constant(&graph, @as(f64, 2));
    const x3 = try constant(&graph, @as(f64, 3));
    const x4 = try constant(&graph, @as(f64, 4));

    const y = try placeholder(&graph, &[_]usize{});
    const y1 = try constant(&graph, @as(f64, 3));
    const y2 = try constant(&graph, @as(f64, 5));
    const y3 = try constant(&graph, @as(f64, 7));
    const y4 = try constant(&graph, @as(f64, 9));

    const y_hat = try add(&graph, try multiply(&graph, m, x), b);
    const loss = try absolute(&graph, try subtract(&graph, y, y_hat));
    const gradients = try gradient(&graph, loss, &[_]Tensor{ m, b });
    const step_size = try constant(&graph, @as(f64, 0.01));
    const dm = try multiply(&graph, gradients[0], step_size);
    const db = try multiply(&graph, gradients[1], step_size);
    const improve_m = try assign(&graph, m, try subtract(&graph, m, dm));
    const improve_b = try assign(&graph, b, try subtract(&graph, b, db));
    var session = try Session.init(allocator, &graph);
    defer session.deinit();

    var environment1 = Environment.init(&session.arena.allocator);
    try environment1.putNoClobber(x, x1);
    try environment1.putNoClobber(y, y1);

    var environment2 = Environment.init(&session.arena.allocator);
    try environment2.putNoClobber(x, x2);
    try environment2.putNoClobber(y, y2);

    var environment3 = Environment.init(&session.arena.allocator);
    try environment3.putNoClobber(x, x3);
    try environment3.putNoClobber(y, y3);

    var environment4 = Environment.init(&session.arena.allocator);
    try environment4.putNoClobber(x, x4);
    try environment4.putNoClobber(y, y4);

    const actual = try session.run(.{
        .tensors = &[_]Tensor{loss},
        .environment = environment1,
    });
    const actual_loss = actual[0];
    expectEqual(f64, actual_loss.f64, try eager.constant(&arena.allocator, @as(f64, 11)));

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        if (i % 4 == 0) {
            _ = try session.run(.{
                .tensors = &[_]Tensor{ improve_m, improve_b },
                .environment = environment4,
            });
        } else if (i % 3 == 0) {
            _ = try session.run(.{
                .tensors = &[_]Tensor{ improve_m, improve_b },
                .environment = environment3,
            });
        } else if (i % 2 == 0) {
            _ = try session.run(.{
                .tensors = &[_]Tensor{ improve_m, improve_b },
                .environment = environment2,
            });
        } else {
            _ = try session.run(.{
                .tensors = &[_]Tensor{ improve_m, improve_b },
                .environment = environment1,
            });
        }
    }

    const actual1 = try session.run(.{
        .tensors = &[_]Tensor{ loss, m, b },
        .environment = environment1,
    });
    const actual_loss1 = actual1[0];
    const actual_m1 = actual1[1];
    const actual_b1 = actual1[2];
    expectEqual(f64, actual_loss1.f64, try eager.constant(&arena.allocator, @as(f64, 2.9999999999895444e-02)));
    expectEqual(f64, actual_m1.f64, try eager.constant(&arena.allocator, @as(f64, 1.9700000000000204e+00)));
    expectEqual(f64, actual_b1.f64, try eager.constant(&arena.allocator, @as(f64, 1.0000000000000844e+00)));
}
