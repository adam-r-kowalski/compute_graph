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
    if (!std.mem.eql(usize, variable.shape, value.shape))
        return error.ShapeMismatch;
    if (variable.scalarType != value.scalarType)
        return error.ScalarTypeMismatch;
    try graph.assigns.append(Assign{
        .variable = variable,
        .value = value,
    });
    return Tensor{
        .tensorType = .{ .assign = graph.assigns.len - 1 },
        .shape = variable.shape,
        .scalarType = variable.scalarType,
    };
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
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
    std.testing.expectEqual(b.scalarType, .f64);
    const c = try constant(&graph, [_][2]f64{
        .{ 1, 1 },
        .{ 1, 1 },
    });
    const d = try add(&graph, b, c);
    const e = try assign(&graph, b, d);
    std.testing.expect(std.mem.eql(usize, e.shape, &[_]usize{ 2, 2 }));
    std.testing.expectEqual(e.scalarType, .f64);
    var session = try Session.init(allocator, &graph);
    defer session.deinit();

    const actual1 = try session.run(&[_]Tensor{ e, b }, .{});
    const expected1 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 2, 3 },
        .{ 4, 5 },
    });
    expectEqual(f64, actual1[0].f64, expected1);
    expectEqual(f64, actual1[1].f64, expected1);

    const actual2 = try session.run(&[_]Tensor{ e, b }, .{});
    const expected2 = try eager.constant(&arena.allocator, [_][2]f64{
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
    expectEqual(f64, actual2[1].f64, expected2);

    const actual3 = try session.run(&[_]Tensor{ e, b }, .{});
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

    const x = try placeholder(&graph, &[_]usize{}, .f64);
    const xs = [_]Tensor{
        try constant(&graph, @as(f64, 0)),
        try constant(&graph, @as(f64, 1)),
        try constant(&graph, @as(f64, 2)),
        try constant(&graph, @as(f64, 3)),
        try constant(&graph, @as(f64, 4)),
    };

    const y = try placeholder(&graph, &[_]usize{}, .f64);
    const ys = [_]Tensor{
        try constant(&graph, @as(f64, 1)),
        try constant(&graph, @as(f64, 3)),
        try constant(&graph, @as(f64, 5)),
        try constant(&graph, @as(f64, 7)),
        try constant(&graph, @as(f64, 9)),
    };

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

    var environments = try session.arena.allocator.alloc(Environment, xs.len);
    for (environments) |*environment, i| {
        environment.* = Environment.init(&session.arena.allocator);
        try environment.putNoClobber(x, xs[i]);
        try environment.putNoClobber(y, ys[i]);
    }

    const actual = try session.run(&[_]Tensor{loss}, .{
        .environment = environments[2],
    });
    const actual_loss = actual[0];
    expectEqual(f64, actual_loss.f64, try eager.constant(&arena.allocator, @as(f64, 17)));

    var i: usize = 0;
    var j: usize = 0;
    while (i < 1000) : (i += 1) {
        _ = try session.run(&[_]Tensor{ improve_m, improve_b }, .{
            .environment = environments[j],
        });
        j = (j + 1) % environments.len;
    }

    const actual1 = try session.run(&[_]Tensor{ loss, m, b }, .{
        .environment = environments[0],
    });
    const actual_loss1 = actual1[0];
    const actual_m1 = actual1[1];
    const actual_b1 = actual1[2];
    expectEqual(f64, actual_m1.f64, try eager.constant(&arena.allocator, @as(f64, 2.02)));
    expectEqual(f64, actual_b1.f64, try eager.constant(&arena.allocator, @as(f64, 1.02)));
    expectEqual(f64, actual_loss1.f64, try eager.constant(&arena.allocator, @as(f64, 0.02)));
}
