const std = @import("std");
const expect = std.testing.expect;
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const ScalarType = tensor.ScalarType;
const eager = @import("../eager.zig");
const CpuTensorUnion = eager.CpuTensorUnion;
const arrayInfo = @import("../util/array_info.zig").arrayInfo;
const expectEqual = @import("../testing.zig").expectEqual;
const Session = @import("session.zig").Session;

fn tensorScalarType(comptime T: type) ScalarType {
    return switch (T) {
        f64 => .f64,
        f32 => .f32,
        f16 => .f16,
        i64 => .i64,
        i32 => .i32,
        i8 => .i8,
        else => @compileError("ScalarType not supported"),
    };
}

pub fn constant(comptime T: type, graph: *Graph, literal: var) !Tensor {
    const eager_tensor = try eager.constant(T, &graph.arena.allocator, literal);
    try graph.constants.append(CpuTensorUnion.init(eager_tensor));
    return Tensor{
        .tensorType = .{ .constant = graph.constants.len - 1 },
        .shape = eager_tensor.shape,
        .scalarType = tensorScalarType(@TypeOf(eager_tensor).ScalarType),
    };
}

test "constant scalar" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f64, &graph, 5);
    const actualString = try std.fmt.allocPrint(&arena.allocator, "{}", .{x});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{x});
    const expected = try eager.constant(f64, &arena.allocator, 5);
    expectEqual(f64, actual[0].f64, expected);
    std.testing.expectEqual(x.shape, &[_]usize{});
    std.testing.expectEqual(x.scalarType, .f64);
    expect(std.mem.eql(u8, actualString, "Tensor(f64)"));
}

test "constant array" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try Graph.init(&arena.allocator);
    const x = try constant(f32, &graph, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    const actualString = try std.fmt.allocPrint(&arena.allocator, "{}", .{x});
    var session = try Session.init(&arena.allocator, &graph);
    const actual = try session.run(&[_]Tensor{x});
    const expected = try eager.constant(f32, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    });
    expectEqual(f32, actual[0].f32, expected);
    std.testing.expect(std.mem.eql(usize, x.shape, &[_]usize{ 3, 2 }));
    std.testing.expectEqual(x.scalarType, .f32);
    expect(std.mem.eql(u8, actualString, "Tensor([3][2]f32)"));
}
