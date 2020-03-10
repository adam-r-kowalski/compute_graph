const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const reduce = @import("reduce.zig").reduce;
const expectEqual = @import("../testing.zig").expectEqual;

fn maximumScalar(comptime T: type) T {
    return switch (T) {
        f64 => std.math.f64_max,
        f32 => std.math.f32_max,
        f16 => std.math.f16_max,
        else => std.math.maxInt(T),
    };
}

pub fn minimum(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), dimension: ?usize) !CpuTensor(T) {
    return try reduce(T, allocator, tensor, dimension, struct {
        fn call(accumulator: T, value: T) T {
            return std.math.min(accumulator, value);
        }
    }.call, maximumScalar(T));
}

test "minimum rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try minimum(f64, &arena.allocator, x, null);
    const expected = try constant(f64, &arena.allocator, -5);
    expectEqual(f64, actual, expected);
}

test "minimum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 5, 10, 7, 8, 10 });
    const actual = try minimum(i32, &arena.allocator, x, null);
    const expected = try constant(i32, &arena.allocator, 5);
    expectEqual(i32, actual, expected);
}

test "minimum rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try minimum(f16, &arena.allocator, x, null);
    const expected = try constant(f16, &arena.allocator, 5);
    expectEqual(f16, actual, expected);
}

test "minimum rank 2 across 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try minimum(f16, &arena.allocator, x, 0);
    const expected = try constant(f16, &arena.allocator, .{ -3, 2 });
    expectEqual(f16, actual, expected);
}

test "minimum rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i8, &arena.allocator, .{
        .{
            .{ 5, 10 },
            .{ 7, 8 },
        },
        .{
            .{ 10, 8 },
            .{ 2, 6 },
        },
    });
    const actual = try minimum(i8, &arena.allocator, x, null);
    const expected = try constant(i8, &arena.allocator, 2);
    expectEqual(i8, actual, expected);
}

test "minimum rank 3 accross 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try minimum(i64, &arena.allocator, x, 0);
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, 2 },
        .{ -3, 4 },
    });
    expectEqual(i64, actual, expected);
}

test "minimum rank 3 accross 1 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try minimum(i64, &arena.allocator, x, 1);
    const expected = try constant(i64, &arena.allocator, .{
        .{ -3, 2 },
        .{ 5, 6 },
    });
    expectEqual(i64, actual, expected);
}

test "minimum rank 3 accross 2 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i64, &arena.allocator, .{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try minimum(i64, &arena.allocator, x, 2);
    const expected = try constant(i64, &arena.allocator, .{
        .{ 1, -3 },
        .{ 5, 7 },
    });
    expectEqual(i64, actual, expected);
}
