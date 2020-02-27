const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

fn addSameShape(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
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

fn addBroadcastScalar(comptime T: type, allocator: *Allocator, scalar: T, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    const array = tensor.storage.array;
    const new_array = try allocator.alloc(T, array.len);
    errdefer allocator.free(new_array);
    for (array) |e, i| new_array[i] = scalar + array[i];
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = .{ .array = new_array },
    };
}

fn broadcastShape(allocator: *Allocator, x: []const usize, y: []const usize) ![]usize {
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

fn addBroadcast(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    const shape = try broadcastShape(allocator, x.shape, y.shape);
    return error.CouldNotBroadcastShapes;
}

pub fn add(comptime T: type, allocator: *Allocator, x: CpuTensor(T), y: CpuTensor(T)) !CpuTensor(T) {
    if (std.mem.eql(usize, x.shape, y.shape))
        return try addSameShape(T, allocator, x, y);
    if (x.shape.len == 0)
        return try addBroadcastScalar(T, allocator, x.storage.scalar, y);
    if (y.shape.len == 0)
        return try addBroadcastScalar(T, allocator, y.storage.scalar, x);
    return try addBroadcast(T, allocator, x, y);
}

test "add rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, 5);
    const y = try constant(f64, &arena.allocator, 10);
    const actual = try add(f64, &arena.allocator, x, y);
    const expected = try constant(f64, &arena.allocator, 15);
    expectEqual(f64, actual, expected);
}

test "add rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try add(i32, &arena.allocator, x, x);
    const expected = try constant(i32, &arena.allocator, .{ 2, -4, 6, -8, -10, 12 });
    expectEqual(i32, actual, expected);
}

test "add rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try add(f16, &arena.allocator, x, x);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 2, -4 },
        .{ 6, -8 },
        .{ -10, 12 },
    });
    expectEqual(f16, actual, expected);
}

test "add rank 3" {
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
    const actual = try add(i8, &arena.allocator, x, x);
    const expected = try constant(i8, &arena.allocator, .{
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

test "add broadcast scalar rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(i32, &arena.allocator, 5);
    const tensor = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try add(i32, &arena.allocator, scalar, tensor);
    const actual2 = try add(i32, &arena.allocator, tensor, scalar);
    const expected = try constant(i32, &arena.allocator, .{ 6, 3, 8, 1, 0, 11 });
    expectEqual(i32, actual, expected);
    expectEqual(i32, actual2, expected);
}

test "add broadcast scalar rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(f16, &arena.allocator, 3);
    const tensor = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try add(f16, &arena.allocator, scalar, tensor);
    const actual2 = try add(f16, &arena.allocator, tensor, scalar);
    const expected = try constant(f16, &arena.allocator, .{
        .{ 4, 1 },
        .{ 6, -1 },
        .{ -2, 9 },
    });
    expectEqual(f16, actual, expected);
    expectEqual(f16, actual2, expected);
}

test "add broadcast scalar rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scalar = try constant(i8, &arena.allocator, -5);
    const tensor = try constant(i8, &arena.allocator, .{
        .{
            .{ 1, -2 },
            .{ 3, -4 },
        },
        .{
            .{ 5, -6 },
            .{ 7, -8 },
        },
    });
    const actual = try add(i8, &arena.allocator, scalar, tensor);
    const actual2 = try add(i8, &arena.allocator, tensor, scalar);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ -4, -7 },
            .{ -2, -9 },
        },
        .{
            .{ 0, -11 },
            .{ 2, -13 },
        },
    });
    expectEqual(i8, actual, expected);
    expectEqual(i8, actual2, expected);
}

pub fn addBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
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
    const x = try constant(f64, &arena.allocator, 4);
    const y = try constant(f64, &arena.allocator, 10);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(f64, &arena.allocator, 1);
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 1, 2, 3, 4, 5 });
    const y = try constant(f64, &arena.allocator, .{ 6, 7, 8, 9, 10 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}

test "add backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const y = try constant(f64, &arena.allocator, .{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try addBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){ x, y },
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    expectEqual(f64, actual[0], expected);
    expectEqual(f64, actual[1], expected);
}
