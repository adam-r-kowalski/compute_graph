const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const linearIndex = cpu_tensor.linearIndex;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

fn newShape(allocator: *Allocator, shape: []const usize, dimension: ?usize) ![]const usize {
    if (dimension) |d| {
        if (d >= shape.len) return error.InvalidDimension;
        const new_shape = try allocator.alloc(usize, shape.len - 1);
        errdefer allocator.free(new_shape);
        for (shape) |s, i| {
            if (i < d) {
                new_shape[i] = shape[i];
            } else if (i > d) {
                new_shape[i - 1] = shape[i];
            }
        }
        return new_shape;
    } else {
        return &[_]usize{};
    }
}

test "newShape null" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, null);
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{}));
}

test "newShape dimension 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, 0);
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 2, 3 }));
}

test "newShape dimension 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, 1);
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 1, 3 }));
}

test "newShape dimension 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, 2);
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 1, 2 }));
}

test "newShape invalid dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    _ = newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, 3) catch |err| switch (err) {
        error.InvalidDimension => {},
        else => unreachable,
    };
}

fn incrementCartesianIndex(shape: []const usize, cartesian_index: []usize) void {
    var i = shape.len - 1;
    var increment = true;
    while (increment) {
        cartesian_index[i] += 1;
        if (cartesian_index[i] == shape[i]) {
            cartesian_index[i] = 0;
            if (i > 0) {
                i -= 1;
            } else {
                increment = false;
            }
        } else {
            increment = false;
        }
    }
}

test "incrementCartesianIndex" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const shape = [_]usize{ 2, 2 };
    var cartesian_index = [_]usize{ 0, 0 };

    incrementCartesianIndex(shape[0..], cartesian_index[0..]);
    std.testing.expect(std.mem.eql(usize, &cartesian_index, &[_]usize{ 0, 1 }));

    incrementCartesianIndex(shape[0..], cartesian_index[0..]);
    std.testing.expect(std.mem.eql(usize, &cartesian_index, &[_]usize{ 1, 0 }));

    incrementCartesianIndex(shape[0..], cartesian_index[0..]);
    std.testing.expect(std.mem.eql(usize, &cartesian_index, &[_]usize{ 1, 1 }));

    incrementCartesianIndex(shape[0..], cartesian_index[0..]);
    std.testing.expect(std.mem.eql(usize, &cartesian_index, &[_]usize{ 0, 0 }));
}

fn maximumCartesianIndex(shape: []const usize, cartesian_index: []const usize) bool {
    var i: usize = 0;
    while (i < shape.len) {
        if (cartesian_index[i] < shape[i] - 1)
            return false;
        i += 1;
    }
    return true;
}

test "maximumCartesianIndex" {
    std.testing.expect(!maximumCartesianIndex(&[_]usize{ 2, 2 }, &[_]usize{ 0, 0 }));
    std.testing.expect(!maximumCartesianIndex(&[_]usize{ 2, 2 }, &[_]usize{ 0, 1 }));
    std.testing.expect(!maximumCartesianIndex(&[_]usize{ 2, 2 }, &[_]usize{ 1, 0 }));
    std.testing.expect(maximumCartesianIndex(&[_]usize{ 2, 2 }, &[_]usize{ 1, 1 }));
}

pub fn sum(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T), dimension: ?usize) !CpuTensor(T) {
    const shape = try newShape(allocator, tensor.shape, dimension);
    errdefer allocator.free(shape);
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = scalar },
            };
        },
        .array => |array| {
            if (dimension) |d| {
                if (shape.len > 0) {
                    const sum_array = try allocator.alloc(T, tensorLength(shape));

                    var sum_cartesian_index = try allocator.alloc(usize, shape.len);
                    errdefer allocator.free(sum_cartesian_index);
                    for (sum_cartesian_index) |*e| e.* = 0;

                    var array_cartesian_index = try allocator.alloc(usize, shape.len + 1);
                    errdefer allocator.free(array_cartesian_index);

                    while (true) {
                        for (array_cartesian_index) |*e, i| {
                            if (i < d) {
                                e.* = sum_cartesian_index[i];
                            } else if (i > d) {
                                e.* = sum_cartesian_index[i - 1];
                            } else {
                                e.* = 0;
                            }
                        }

                        var total: T = 0;
                        var i: usize = 0;
                        while (i < tensor.shape[d]) {
                            array_cartesian_index[d] = i;
                            const array_linear_index = linearIndex(tensor.stride, array_cartesian_index);
                            total += array[array_linear_index];
                            i += 1;
                        }
                        const sum_linear_index = linearIndex(stride, sum_cartesian_index);
                        sum_array[sum_linear_index] = total;

                        if (maximumCartesianIndex(shape, sum_cartesian_index)) break;
                        incrementCartesianIndex(shape, sum_cartesian_index);
                    }

                    return CpuTensor(T){
                        .shape = shape,
                        .stride = stride,
                        .storage = .{ .array = sum_array },
                    };
                }
            }

            var total: T = 0;
            for (array) |e| total += e;
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = total },
            };
        },
    }
}

test "sum rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, @as(f64, -5));
    const actual = try sum(f64, &arena.allocator, x, null);
    const expected = try constant(&arena.allocator, @as(f64, -5));
    expectEqual(f64, actual, expected);
}

test "sum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{ 5, 10, 7, 8, 10 });
    const actual = try sum(i32, &arena.allocator, x, null);
    const expected = try constant(&arena.allocator, @as(i32, 40));
    expectEqual(i32, actual, expected);
}

test "sum rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 5, 10 },
        .{ 7, 8 },
        .{ 10, 8 },
    });
    const actual = try sum(f16, &arena.allocator, x, null);
    const expected = try constant(&arena.allocator, @as(f16, 48));
    expectEqual(f16, actual, expected);
}

test "sum rank 2 across 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2]f16{
        .{ 1, 2 },
        .{ -3, 4 },
        .{ 5, 6 },
    });
    const actual = try sum(f16, &arena.allocator, x, 0);
    const expected = try constant(&arena.allocator, [_]f16{ 3, 12 });
    expectEqual(f16, actual, expected);
}

test "sum rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i8{
        .{
            .{ 5, 10 },
            .{ 7, 8 },
        },
        .{
            .{ 10, 8 },
            .{ 2, 6 },
        },
    });
    const actual = try sum(i8, &arena.allocator, x, null);
    const expected = try constant(&arena.allocator, @as(i8, 56));
    expectEqual(i8, actual, expected);
}

test "sum rank 3 accross 0 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i64{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try sum(i64, &arena.allocator, x, 0);
    const expected = try constant(&arena.allocator, [_][2]i64{
        .{ 6, 8 },
        .{ 4, 12 },
    });
    expectEqual(i64, actual, expected);
}

test "sum rank 3 accross 1 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i64{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try sum(i64, &arena.allocator, x, 1);
    const expected = try constant(&arena.allocator, [_][2]i64{
        .{ -2, 6 },
        .{ 12, 14 },
    });
    expectEqual(i64, actual, expected);
}

test "sum rank 3 accross 2 dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_][2][2]i64{
        .{
            .{ 1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ 5, 6 },
            .{ 7, 8 },
        },
    });
    const actual = try sum(i64, &arena.allocator, x, 2);
    const expected = try constant(&arena.allocator, [_][2]i64{
        .{ 3, 1 },
        .{ 11, 15 },
    });
    expectEqual(i64, actual, expected);
}

pub fn sumBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    const shape = input.shape;
    const stride = input.stride;
    const gradient = context.gradient_input.storage.scalar;

    switch (input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = gradient },
            };
        },
        .array => |array| {
            var new_array = try context.allocator.alloc(T, array.len);
            for (new_array) |*e, i| e.* = gradient;
            outputs[0] = CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "sum backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(&arena.allocator, @as(f64, 4));
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try sumBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
    });
    const expected = try constant(&arena.allocator, @as(f64, 1));
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(&arena.allocator, [_]f64{ 1, 2, 3, 4, 5 });
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try sumBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
    });
    const expected = try constant(&arena.allocator, [_]f64{ 1, 1, 1, 1, 1 });
    expectEqual(f64, actual[0], expected);
}

test "sum backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const forward_input = try constant(&arena.allocator, [_][2]f64{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const gradient_input = try constant(&arena.allocator, @as(f64, 1));
    const actual = try sumBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){forward_input},
    });
    const expected = try constant(&arena.allocator, [_][2]f64{
        .{ 1, 1 },
        .{ 1, 1 },
    });
    expectEqual(f64, actual[0], expected);
}
