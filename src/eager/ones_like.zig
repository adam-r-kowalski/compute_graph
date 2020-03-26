const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const copy = @import("cpu_tensor.zig").copy;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn onesLike(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = try copy(usize, allocator, tensor.shape);
    errdefer allocator.free(shape);
    const stride = try copy(usize, allocator, tensor.stride);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = @as(T, 1) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = @as(T, 1);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "onesLike rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try onesLike(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual, expected);
}

test "onesLike rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try onesLike(i32, &arena.allocator, x);
    const expected = try constant(i32, &arena.allocator, .{ 1, 1, 1, 1, 1, 1 });
    expectEqual(i32, actual, expected);
}

test "onesLike rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try onesLike(f16, &arena.allocator, x);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 1, 1 },
        .{ 1, 1 },
        .{ 1, 1 },
    });
    expectEqual(f16, actual, expected);
}

test "onesLike rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const actual = try onesLike(i8, &arena.allocator, x);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, 1 },
            .{ 1, 1 },
        },
        .{
            .{ 1, 1 },
            .{ 1, 1 },
        },
    });
    expectEqual(i8, actual, expected);
}

test "onesLike rank 1 seperate lifetime" {
    var leak_allocator = std.testing.LeakCountAllocator.init(std.heap.page_allocator);
    defer leak_allocator.validate() catch unreachable;
    const x = try constant(i32, &leak_allocator.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try onesLike(i32, &leak_allocator.allocator, x);
    defer actual.deinit(&leak_allocator.allocator);
    x.deinit(&leak_allocator.allocator);
    const expected = try constant(i32, &leak_allocator.allocator, .{ 1, 1, 1, 1, 1, 1 });
    defer expected.deinit(&leak_allocator.allocator);
    expectEqual(i32, actual, expected);
}
