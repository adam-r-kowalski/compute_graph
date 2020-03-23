const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;
const Session = @import("session.zig").Session;
const expectEqual = @import("../testing.zig").expectEqual;
const add = @import("add.zig").add;
const constant = @import("constant.zig").constant;
const variable = @import("variable.zig").variable;
const eager = @import("../eager.zig");
const absolute = @import("absolute.zig").absolute;
const multiply = @import("multiply.zig").multiply;
const subtract = @import("subtract.zig").subtract;
const placeholder = @import("placeholder.zig").placeholder;
const gradient = @import("gradient.zig").gradient;
const Environment = @import("session.zig").Environment;

pub const Assign = struct {
    variable: Tensor,
    value: Tensor,
};

pub fn assign(graph: *Graph, target: Tensor, value: Tensor) !Tensor {
    if (!std.mem.eql(usize, target.shape, value.shape))
        return error.ShapeMismatch;
    if (target.scalarType != value.scalarType)
        return error.ScalarTypeMismatch;
    try graph.assigns.append(Assign{
        .variable = target,
        .value = value,
    });
    return Tensor{
        .tensorType = .{ .assign = graph.assigns.len - 1 },
        .shape = target.shape,
        .scalarType = target.scalarType,
    };
}

test "assign" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const a = try constant(f64, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const b = try variable(&graph, a);
    std.testing.expect(std.mem.eql(usize, b.shape, &[_]usize{ 2, 2 }));
    std.testing.expectEqual(b.scalarType, .f64);
    const c = try constant(f64, &graph, .{
        .{ 1, 1 },
        .{ 1, 1 },
    });
    const d = try add(&graph, b, c);
    const e = try assign(&graph, b, d);
    std.testing.expect(std.mem.eql(usize, e.shape, &[_]usize{ 2, 2 }));
    std.testing.expectEqual(e.scalarType, .f64);
    var session = Session.init(allocator, &graph);
    defer session.deinit();

    const actual1 = try session.run(.{ e, b });
    const expected1 = try eager.constant(f64, &arena.allocator, .{
        .{ 2, 3 },
        .{ 4, 5 },
    });
    expectEqual(f64, actual1[0].f64, expected1);
    expectEqual(f64, actual1[1].f64, expected1);

    const actual2 = try session.run(.{ e, b });
    const expected2 = try eager.constant(f64, &arena.allocator, .{
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
    expectEqual(f64, actual2[1].f64, expected2);

    const actual3 = try session.run(.{ e, b });
    const expected3 = try eager.constant(f64, &arena.allocator, .{
        .{ 4, 5 },
        .{ 6, 7 },
    });
    expectEqual(f64, actual3[0].f64, expected3);
    expectEqual(f64, actual3[1].f64, expected3);
}

test "linear regression" {
    const allocator = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try variable(&graph, try constant(f64, &graph, 8));
    const b = try variable(&graph, try constant(f64, &graph, 6));

    const x = try placeholder(&graph, &[_]usize{}, .f64);
    const xs = [_]Tensor{
        try constant(f64, &graph, 0),
        try constant(f64, &graph, 1),
        try constant(f64, &graph, 2),
        try constant(f64, &graph, 3),
        try constant(f64, &graph, 4),
    };

    const y = try placeholder(&graph, &[_]usize{}, .f64);
    const ys = [_]Tensor{
        try constant(f64, &graph, 1),
        try constant(f64, &graph, 3),
        try constant(f64, &graph, 5),
        try constant(f64, &graph, 7),
        try constant(f64, &graph, 9),
    };

    const y_hat = try add(&graph, try multiply(&graph, m, x), b);
    const loss = try absolute(&graph, try subtract(&graph, y, y_hat));
    const gradients = try gradient(&graph, loss, &[_]Tensor{ m, b });
    const step_size = try constant(f64, &graph, 0.01);
    const dm = try multiply(&graph, gradients[0], step_size);
    const db = try multiply(&graph, gradients[1], step_size);
    const improve_m = try assign(&graph, m, try subtract(&graph, m, dm));
    const improve_b = try assign(&graph, b, try subtract(&graph, b, db));
    var session = Session.init(allocator, &graph);
    defer session.deinit();

    var environments = try arena.allocator.alloc(Environment, xs.len);
    for (environments) |*environment, i| {
        environment.* = Environment.init(&arena.allocator);
        try environment.putNoClobber(x, xs[i]);
        try environment.putNoClobber(y, ys[i]);
    }

    const actual = try session.run(.{
        .tensors = &[_]Tensor{loss},
        .environment = environments[2],
    });
    const actual_loss = actual[0];
    expectEqual(f64, actual_loss.f64, try eager.constant(f64, &arena.allocator, 17));

    var i: usize = 0;
    var j: usize = 0;
    while (i < 1000) : (i += 1) {
        _ = try session.run(.{
            .tensors = &[_]Tensor{ improve_m, improve_b },
            .environment = environments[j],
        });
        j = (j + 1) % environments.len;
    }

    const actual1 = try session.run(.{
        .tensors = &[_]Tensor{ loss, m, b },
        .environment = environments[0],
    });
    const actual_loss1 = actual1[0];
    const actual_m1 = actual1[1];
    const actual_b1 = actual1[2];
    expectEqual(f64, actual_m1.f64, try eager.constant(f64, &arena.allocator, 2.02));
    expectEqual(f64, actual_b1.f64, try eager.constant(f64, &arena.allocator, 1.02));
    expectEqual(f64, actual_loss1.f64, try eager.constant(f64, &arena.allocator, 0.02));
}
