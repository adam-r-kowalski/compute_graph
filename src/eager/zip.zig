const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_tensor = @import("cpu_tensor.zig");
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const linearIndex = cpu_tensor.linearIndex;
const CpuTensor = cpu_tensor.CpuTensor;
const broadcast = @import("broadcast.zig");
const broadcastShape = broadcast.broadcastShape;
const debroadcastIndex = broadcast.debroadcastIndex;
const maximumCartesianIndex = broadcast.maximumCartesianIndex;
const incrementCartesianIndex = broadcast.incrementCartesianIndex;

fn zipSameShape(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T), f: fn (T, T) T) !CpuTensor(T) {
    const shape = try allocator.alloc(usize, x.shape.len);
    errdefer allocator.free(shape);
    std.mem.copy(usize, shape, x.shape);

    const stride = try allocator.alloc(usize, x.stride.len);
    errdefer allocator.free(stride);
    std.mem.copy(usize, stride, x.stride);

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
    const shape = try allocator.alloc(usize, tensor.shape.len);
    errdefer allocator.free(shape);
    std.mem.copy(usize, shape, tensor.shape);

    const stride = try allocator.alloc(usize, tensor.stride.len);
    errdefer allocator.free(stride);
    std.mem.copy(usize, stride, tensor.stride);

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

fn zipBroadcastScalarFlip(comptime T: type, allocator: *Allocator, scalar: T, tensor: CpuTensor(T), f: fn (T, T) T) !CpuTensor(T) {
    const shape = try allocator.alloc(usize, tensor.shape.len);
    errdefer allocator.free(shape);
    std.mem.copy(usize, shape, tensor.shape);

    const stride = try allocator.alloc(usize, tensor.stride.len);
    errdefer allocator.free(stride);
    std.mem.copy(usize, stride, tensor.stride);

    const array = tensor.storage.array;
    const new_array = try allocator.alloc(T, array.len);
    errdefer allocator.free(new_array);
    for (array) |e, i| new_array[i] = f(array[i], scalar);
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
    defer allocator.free(cartesian_index);
    for (cartesian_index) |*e| e.* = 0;
    const x_cartesian_index = try allocator.alloc(usize, x.shape.len);
    defer allocator.free(x_cartesian_index);
    const y_cartesian_index = try allocator.alloc(usize, y.shape.len);
    defer allocator.free(y_cartesian_index);
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
    if (y.shape.len == 0) {
        return try zipBroadcastScalarFlip(T, allocator, y.storage.scalar, x, f);
    }
    return try zipBroadcast(T, allocator, x, y, f);
}
