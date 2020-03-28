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
const sigmoid = @import("sigmoid.zig").sigmoid;
const meanAbsoluteError = @import("mean_absolute_error.zig").meanAbsoluteError;
const binaryCrossEntropy = @import("binary_cross_entropy.zig").binaryCrossEntropy;
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
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const allocator = &leak_allocator.allocator;
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
    const expected1 = try eager.constant(f64, allocator, .{
        .{ 2, 3 },
        .{ 4, 5 },
    });
    expectEqual(f64, actual1[0].f64, expected1);
    expectEqual(f64, actual1[1].f64, expected1);
    expected1.deinit(allocator);
    for (actual1) |tensor| tensor.deinit(allocator);
    allocator.free(actual1);

    const actual2 = try session.run(.{ e, b });
    const expected2 = try eager.constant(f64, allocator, .{
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f64, actual2[0].f64, expected2);
    expectEqual(f64, actual2[1].f64, expected2);
    expected2.deinit(allocator);
    for (actual2) |tensor| tensor.deinit(allocator);
    allocator.free(actual2);

    const actual3 = try session.run(.{ e, b });
    const expected3 = try eager.constant(f64, allocator, .{
        .{ 4, 5 },
        .{ 6, 7 },
    });
    expectEqual(f64, actual3[0].f64, expected3);
    expectEqual(f64, actual3[1].f64, expected3);
    expected3.deinit(allocator);
    for (actual3) |tensor| tensor.deinit(allocator);
    allocator.free(actual3);
}

test "linear regression" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const allocator = &leak_allocator.allocator;
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
    const loss = try meanAbsoluteError(&graph, y, y_hat);
    const gradients = try gradient(&graph, loss, &[_]Tensor{ m, b });
    const step_size = try constant(f64, &graph, 0.01);
    const dm = try multiply(&graph, gradients[0], step_size);
    const db = try multiply(&graph, gradients[1], step_size);
    const improve_m = try assign(&graph, m, try subtract(&graph, m, dm));
    const improve_b = try assign(&graph, b, try subtract(&graph, b, db));
    var session = Session.init(allocator, &graph);
    defer session.deinit();

    var environments = try allocator.alloc(Environment, xs.len);
    defer {
        for (environments) |environment| environment.deinit();
        allocator.free(environments);
    }
    for (environments) |*environment, i| {
        environment.* = Environment.init(allocator);
        try environment.putNoClobber(x, xs[i]);
        try environment.putNoClobber(y, ys[i]);
    }

    const actual = try session.run(.{
        .tensors = &[_]Tensor{loss},
        .environment = environments[2],
    });
    const actual_loss = actual[0];
    const expected_loss = try eager.constant(f64, allocator, 17);
    expectEqual(f64, actual_loss.f64, expected_loss);
    expected_loss.deinit(allocator);
    for (actual) |tensor| tensor.deinit(allocator);
    allocator.free(actual);

    var i: usize = 0;
    var j: usize = 0;
    while (i < 1000) : (i += 1) {
        const result = try session.run(.{
            .tensors = &[_]Tensor{ improve_m, improve_b },
            .environment = environments[j],
        });
        for (result) |tensor| tensor.deinit(allocator);
        allocator.free(result);
        j = (j + 1) % environments.len;
    }

    const actual1 = try session.run(.{
        .tensors = &[_]Tensor{ loss, m, b },
        .environment = environments[0],
    });
    defer {
        for (actual1) |tensor| tensor.deinit(allocator);
        allocator.free(actual1);
    }
    const expected_loss1 = try eager.constant(f64, allocator, 0.02);
    defer expected_loss1.deinit(allocator);
    const expected_m1 = try eager.constant(f64, allocator, 2.02);
    defer expected_m1.deinit(allocator);
    const expected_b1 = try eager.constant(f64, allocator, 1.02);
    defer expected_b1.deinit(allocator);
    expectEqual(f64, actual1[0].f64, expected_loss1);
    expectEqual(f64, actual1[1].f64, expected_m1);
    expectEqual(f64, actual1[2].f64, expected_b1);
}

test "logistic regression" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const allocator = &leak_allocator.allocator;
    var graph = try Graph.init(allocator);
    defer graph.deinit();
    const m = try variable(&graph, try constant(f64, &graph, -0.5));
    const b = try variable(&graph, try constant(f64, &graph, 0.3));

    const x = try placeholder(&graph, &[_]usize{}, .f64);
    const xs = [_]Tensor{
        try constant(f64, &graph, .{ 0, 5, 3 }),
        try constant(f64, &graph, .{ 2, 7, 8 }),
        try constant(f64, &graph, .{ 1, 4, 6 }),
        try constant(f64, &graph, .{ 7, 9, 4 }),
    };

    const y = try placeholder(&graph, &[_]usize{}, .f64);
    const ys = [_]Tensor{
        try constant(f64, &graph, .{ 0, 1, 0 }),
        try constant(f64, &graph, .{ 0, 1, 1 }),
        try constant(f64, &graph, .{ 0, 0, 1 }),
        try constant(f64, &graph, .{ 1, 1, 0 }),
    };

    const y_hat = try sigmoid(&graph, try add(&graph, try multiply(&graph, m, x), b));
    const loss = try binaryCrossEntropy(&graph, y, y_hat);
    const gradients = try gradient(&graph, loss, &[_]Tensor{ m, b });
    const step_size = try constant(f64, &graph, 0.01);
    const dm = try multiply(&graph, gradients[0], step_size);
    const db = try multiply(&graph, gradients[1], step_size);
    const improve_m = try assign(&graph, m, try subtract(&graph, m, dm));
    const improve_b = try assign(&graph, b, try subtract(&graph, b, db));
    var session = Session.init(allocator, &graph);
    defer session.deinit();

    var environments = try allocator.alloc(Environment, xs.len);
    defer {
        for (environments) |environment| environment.deinit();
        allocator.free(environments);
    }
    for (environments) |*environment, i| {
        environment.* = Environment.init(allocator);
        try environment.putNoClobber(x, xs[i]);
        try environment.putNoClobber(y, ys[i]);
    }

    const actual = try session.run(.{
        .tensors = &[_]Tensor{loss},
        .environment = environments[2],
    });
    const expected_loss = try eager.constant(f64, allocator, 1.1769);
    expectEqual(f64, actual[0].f64, expected_loss);
    expected_loss.deinit(allocator);
    for (actual) |tensor| tensor.deinit(allocator);
    allocator.free(actual);

    var i: usize = 0;
    var j: usize = 0;
    while (i < 1000) : (i += 1) {
        const result = try session.run(.{
            .tensors = &[_]Tensor{ improve_m, improve_b },
            .environment = environments[j],
        });
        for (result) |tensor| tensor.deinit(allocator);
        allocator.free(result);
        j = (j + 1) % environments.len;
    }

    const actual1 = try session.run(.{
        .tensors = &[_]Tensor{ loss, m, b, y_hat },
        .environment = environments[0],
    });
    defer {
        for (actual1) |tensor| tensor.deinit(allocator);
        allocator.free(actual1);
    }
    const expected_loss1 = try eager.constant(f64, allocator, 0.4443);
    defer expected_loss1.deinit(allocator);
    const expected_m1 = try eager.constant(f64, allocator, 0.3835);
    defer expected_m1.deinit(allocator);
    const expected_b1 = try eager.constant(f64, allocator, -1.1908);
    defer expected_b1.deinit(allocator);
    const expected_y_hat1 = try eager.constant(f64, allocator, .{
        0.2331, 0.6741, 0.4899,
    });
    defer expected_y_hat1.deinit(allocator);
    expectEqual(f64, actual1[0].f64, expected_loss1);
    expectEqual(f64, actual1[1].f64, expected_m1);
    expectEqual(f64, actual1[2].f64, expected_b1);
    expectEqual(f64, actual1[3].f64, expected_y_hat1);
}
