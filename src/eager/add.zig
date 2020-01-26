const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn add(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    if (!std.mem.eql(usize, x.shape, y.shape))
        return error.ShapeMismatch;
    const shape = x.shape;
    const stride = x.stride;
    switch (x.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = scalar + y.storage.scalar },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            const y_array = y.storage.array;
            for (array) |e, i| new_array[i] = e + y_array[i];
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "add rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, 5));
    const y = try constant(&arena.allocator, @as(f64, 10));
    const actual = try add(f64, &arena.allocator, x, y);
    const expected = try constant(&arena.allocator, @as(f64, 15));
    expectEqual(f64, actual, expected);
}

test "add rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{ 1, -2, 3, -4, -5, 6 });
    const actual = try add(i32, &arena.allocator, x, x);
    const expected = try constant(&arena.allocator, [_]i32{ 2, -4, 6, -8, -10, 12 });
    expectEqual(i32, actual, expected);
}

test "add rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try add(f16, &arena.allocator, x, x);
    const expected = try constant(&arena.allocator, [_][2]f16{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f16, actual, expected);
}

test "add rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const actual = try add(i8, &arena.allocator, x, x);
    const expected = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 2, -4 },
            .{ 6, -8 },
        },
        .{
            .{ 10, -12 },
            .{ 14, -16 },
        },
    });
    expectEqual(i8, actual, expected);
}

pub fn add_backward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 2);
    const outputs = try context.allocator.alloc(CpuTensor(T), 2);
    errdefer context.allocator.free(outputs);
    outputs[0] = context.gradient_input;
    outputs[1] = context.gradient_input;
    return outputs;
}

test "add backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, 4));
    const y = try constant(&arena.allocator, @as(f64, 10));
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try add_backward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(&arena.allocator, @as(f64, 1));
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]f64{ 1, 2, 3, 4, 5 });
    const y = try constant(&arena.allocator, [_]f64{ 6, 7, 8, 9, 10 });
    const gradient_input = try constant(&arena.allocator, [_]f64{ 2, 4, 6, 8, 10 });
    const actual = try add_backward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(&arena.allocator, [_]f64{ 2, 4, 6, 8, 10 });
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(&arena.allocator, [_][2]f64{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const gradient_input = try constant(&arena.allocator, [_][2]f64{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try add_backward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}
