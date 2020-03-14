const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const linearIndex = cpu_tensor.linearIndex;
const broadcast = @import("broadcast.zig");
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;
const zeroBroadcastedIndex = broadcast.zeroBroadcastedIndex;

pub const ReduceParameters = struct {
    dimension: ?usize = null,
    keep_dimensions: bool = false,
};

pub fn newShape(allocator: *Allocator, shape: []const usize, parameters: ReduceParameters) ![]const usize {
    if (parameters.dimension) |d| {
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
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, .{});
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{}));
}

test "newShape dimension 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, .{ .dimension = 0 });
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 2, 3 }));
}

test "newShape dimension 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, .{ .dimension = 1 });
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 1, 3 }));
}

test "newShape dimension 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const actual = try newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, .{ .dimension = 2 });
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 1, 2 }));
}

test "newShape invalid dimension" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    _ = newShape(&arena.allocator, &[_]usize{ 1, 2, 3 }, .{ .dimension = 3 }) catch |err| switch (err) {
        error.InvalidDimension => {},
        else => unreachable,
    };
}

fn reduceAcrossDimension(
    comptime T: type,
    allocator: *Allocator,
    tensor: CpuTensor(T),
    dimension: usize,
    array: []const T,
    shape: []const usize,
    stride: []const usize,
    reducer: fn (T, T) T,
    identity: T,
) !CpuTensor(T) {
    const reduce_array = try allocator.alloc(T, tensorLength(shape));
    errdefer allocator.free(reduce_array);

    var reduce_cartesian_index = try allocator.alloc(usize, shape.len);
    defer allocator.free(reduce_cartesian_index);
    for (reduce_cartesian_index) |*e| e.* = 0;

    var array_cartesian_index = try allocator.alloc(usize, shape.len + 1);
    defer allocator.free(array_cartesian_index);

    while (true) {
        zeroBroadcastedIndex(reduce_cartesian_index, dimension, array_cartesian_index);

        var accumulator = identity;
        var i: usize = 0;
        while (i < tensor.shape[dimension]) {
            array_cartesian_index[dimension] = i;
            const array_linear_index = linearIndex(tensor.stride, array_cartesian_index);
            accumulator = reducer(accumulator, array[array_linear_index]);
            i += 1;
        }
        const reduce_linear_index = linearIndex(stride, reduce_cartesian_index);
        reduce_array[reduce_linear_index] = accumulator;

        if (maximumCartesianIndex(shape, reduce_cartesian_index)) break;
        incrementCartesianIndex(shape, reduce_cartesian_index);
    }

    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = reduce_array },
    };
}

pub fn reduce(
    comptime T: type,
    allocator: *Allocator,
    tensor: CpuTensor(T),
    reducer: fn (T, T) T,
    identity: T,
    parameters: ReduceParameters,
) !CpuTensor(T) {
    const shape = try newShape(allocator, tensor.shape, parameters);
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
            if (parameters.dimension) |d|
                if (shape.len > 0)
                    return reduceAcrossDimension(T, allocator, tensor, d, array, shape, stride, reducer, identity);

            var accumulator = identity;
            for (array) |e| accumulator = reducer(accumulator, e);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = accumulator },
            };
        },
    }
}
