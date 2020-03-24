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

pub fn zeroBroadcastedIndex(cartesian_index: []const usize, dimension: usize, broadcasted_cartesian_index: []usize) void {
    for (broadcasted_cartesian_index) |*e, i| {
        if (i < dimension) {
            e.* = cartesian_index[i];
        } else if (i > dimension) {
            e.* = cartesian_index[i - 1];
        } else {
            e.* = 0;
        }
    }
}

pub fn zeroBroadcastedIndexKeepDimension(cartesian_index: []const usize, dimension: usize, broadcasted_cartesian_index: []usize) void {
    for (broadcasted_cartesian_index) |*e, i|
        e.* = if (i < dimension or i > dimension) cartesian_index[i] else 0;
}
