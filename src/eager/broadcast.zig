const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const linearIndex = cpu_tensor.linearIndex;

pub fn incrementCartesianIndex(shape: []const usize, cartesian_index: []usize) void {
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

pub fn maximumCartesianIndex(shape: []const usize, cartesian_index: []const usize) bool {
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

pub fn debroadcastIndex(shape: []const usize, broadcastIndex: []const usize, outputIndex: []usize) void {
    std.debug.assert(shape.len == outputIndex.len);
    const delta = broadcastIndex.len - outputIndex.len;
    var i: usize = 0;
    while (i < outputIndex.len) : (i += 1) {
        outputIndex[i] = if (shape[i] == 1) 0 else broadcastIndex[i + delta];
    }
}

test "debroadcast index rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const shape = &[_]usize{};
    const broadcastIndex = &[_]usize{};
    const index = try arena.allocator.alloc(usize, shape.len);
    debroadcastIndex(shape, broadcastIndex, index);
    std.testing.expect(std.mem.eql(usize, index, &[_]usize{}));
}

test "debroadcast index rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const shape = &[_]usize{2};
    const broadcastIndex = &[_]usize{ 1, 2, 1 };
    const index = try arena.allocator.alloc(usize, shape.len);
    debroadcastIndex(shape, broadcastIndex, index);
    std.testing.expect(std.mem.eql(usize, index, &[_]usize{1}));
}

test "debroadcast index rank 1 example 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const shape = &[_]usize{2};
    const broadcastIndex = &[_]usize{ 1, 2, 0 };
    const index = try arena.allocator.alloc(usize, shape.len);
    debroadcastIndex(shape, broadcastIndex, index);
    std.testing.expect(std.mem.eql(usize, index, &[_]usize{0}));
}

test "debroadcast index rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const shape = &[_]usize{ 2, 2 };
    const broadcastIndex = &[_]usize{ 1, 1, 0 };
    const index = try arena.allocator.alloc(usize, shape.len);
    debroadcastIndex(shape, broadcastIndex, index);
    std.testing.expect(std.mem.eql(usize, index, &[_]usize{ 1, 0 }));
}

pub fn broadcastShape(allocator: *Allocator, x: []const usize, y: []const usize) ![]usize {
    const len = std.math.max(x.len, y.len);
    const shape = try allocator.alloc(usize, len);

    const candidate = struct {
        fn closure(s: []const usize, i: usize) usize {
            return if (i <= s.len) s[s.len - i] else 1;
        }
    }.closure;

    var i: usize = 1;
    while (i <= len) : (i += 1) {
        const x_candidate = candidate(x, i);
        const y_candidate = candidate(y, i);
        if (x_candidate == y_candidate) {
            shape[len - i] = x_candidate;
        } else if (x_candidate == 1) {
            shape[len - i] = y_candidate;
        } else if (y_candidate == 1) {
            shape[len - i] = x_candidate;
        } else {
            return error.CouldNotBroadcastShapes;
        }
    }

    return shape;
}

test "broadcast shape" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const actual = try broadcastShape(&arena.allocator, &[_]usize{ 8, 1, 6, 1 }, &[_]usize{ 7, 1, 5 });
    std.testing.expect(std.mem.eql(usize, actual, &[_]usize{ 8, 7, 6, 5 }));

    const actual2 = try broadcastShape(&arena.allocator, &[_]usize{ 5, 4 }, &[_]usize{1});
    std.testing.expect(std.mem.eql(usize, actual2, &[_]usize{ 5, 4 }));

    const actual3 = try broadcastShape(&arena.allocator, &[_]usize{ 5, 4 }, &[_]usize{4});
    std.testing.expect(std.mem.eql(usize, actual3, &[_]usize{ 5, 4 }));

    const actual4 = try broadcastShape(&arena.allocator, &[_]usize{ 15, 3, 5 }, &[_]usize{ 15, 1, 5 });
    std.testing.expect(std.mem.eql(usize, actual4, &[_]usize{ 15, 3, 5 }));

    const actual5 = try broadcastShape(&arena.allocator, &[_]usize{ 15, 3, 5 }, &[_]usize{ 3, 5 });
    std.testing.expect(std.mem.eql(usize, actual5, &[_]usize{ 15, 3, 5 }));

    const actual6 = try broadcastShape(&arena.allocator, &[_]usize{ 15, 3, 5 }, &[_]usize{ 3, 1 });
    std.testing.expect(std.mem.eql(usize, actual6, &[_]usize{ 15, 3, 5 }));

    _ = broadcastShape(&arena.allocator, &[_]usize{3}, &[_]usize{4}) catch |err| switch (err) {
        error.CouldNotBroadcastShapes => {},
        else => unreachable,
    };

    _ = broadcastShape(&arena.allocator, &[_]usize{ 2, 1 }, &[_]usize{ 8, 4, 3 }) catch |err| switch (err) {
        error.CouldNotBroadcastShapes => {},
        else => unreachable,
    };
}

fn zipSameShape(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T), f: fn (T, T) T) !CpuTensor(T) {
    const shape = x.shape;
    const stride = x.stride;
    switch (x.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = f(scalar, y.storage.scalar) },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            const y_array = y.storage.array;
            for (array) |e, i| new_array[i] = f(e, y_array[i]);
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

fn zipBroadcastScalar(comptime T: type, allocator: *Allocator, scalar: T, tensor: CpuTensor(T), f: fn (T, T) T) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    const array = tensor.storage.array;
    const new_array = try allocator.alloc(T, array.len);
    errdefer allocator.free(new_array);
    for (array) |e, i| new_array[i] = f(scalar, array[i]);
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = new_array },
    };
}

fn zipBroadcast(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T), f: fn (T, T) T) !CpuTensor(T) {
    const shape = try broadcastShape(allocator, x.shape, y.shape);
    errdefer allocator.free(shape);
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    const cartesian_index = try allocator.alloc(usize, shape.len);
    errdefer allocator.free(cartesian_index);
    for (cartesian_index) |*e| e.* = 0;
    const x_cartesian_index = try allocator.alloc(usize, x.shape.len);
    errdefer allocator.free(x_cartesian_index);
    const y_cartesian_index = try allocator.alloc(usize, y.shape.len);
    errdefer allocator.free(y_cartesian_index);
    const array = try allocator.alloc(T, tensorLength(shape));
    errdefer allocator.free(array);
    const x_array = x.storage.array;
    const y_array = y.storage.array;
    while (true) {
        debroadcastIndex(x.shape, cartesian_index, x_cartesian_index);
        debroadcastIndex(y.shape, cartesian_index, y_cartesian_index);
        const x_index = linearIndex(x.stride, x_cartesian_index);
        const y_index = linearIndex(y.stride, y_cartesian_index);
        const index = linearIndex(stride, cartesian_index);
        array[index] = f(x_array[x_index], y_array[y_index]);
        if (maximumCartesianIndex(shape, cartesian_index)) break;
        incrementCartesianIndex(shape, cartesian_index);
    }
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = array },
    };
}

pub fn zip(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T), f: fn (T, T) T) !CpuTensor(T) {
    if (std.mem.eql(usize, x.shape, y.shape))
        return try zipSameShape(T, allocator, x, y, f);
    if (x.shape.len == 0)
        return try zipBroadcastScalar(T, allocator, x.storage.scalar, y, f);
    if (y.shape.len == 0)
        return try zipBroadcastScalar(T, allocator, y.storage.scalar, x, f);
    return try zipBroadcast(T, allocator, x, y, f);
}
