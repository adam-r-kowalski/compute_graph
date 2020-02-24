const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const CpuTensor = @import("cpu_tensor.zig").CpuTensor;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn negate(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = tensor.shape;
    const stride = tensor.stride;
    switch (tensor.storage) {
        .scalar => |scalar| {
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .scalar = -scalar },
            };
        },
        .array => |array| {
            const new_array = try allocator.alloc(T, array.len);
            errdefer allocator.free(new_array);
            for (array) |e, i| new_array[i] = -e;
            return CpuTensor(T){
                .shape = shape,
                .stride = stride,
                .storage = .{ .array = new_array },
            };
        },
    }
}

test "negate rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -5);
    const actual = try negate(f64, &arena.allocator, x);
    const expected = try constant(f64, &arena.allocator, 5);
    expectEqual(f64, actual, expected);
}

test "negate rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(i32, &arena.allocator, .{ 1, -2, 3, -4, -5, 6 });
    const actual = try negate(i32, &arena.allocator, x);
    const expected = try constant(i32, &arena.allocator, .{ -1, 2, -3, 4, 5, -6 });
    expectEqual(i32, actual, expected);
}

test "negate rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f16, &arena.allocator, .{
        .{ 1, -2 },
        .{ 3, -4 },
        .{ -5, 6 },
    });
    const actual = try negate(f16, &arena.allocator, x);
    const expected = try constant(f16, &arena.allocator, .{
        .{ -1, 2 },
        .{ -3, 4 },
        .{ 5, -6 },
    });
    expectEqual(f16, actual, expected);
}

test "negate rank 3" {
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
    const actual = try negate(i8, &arena.allocator, x);
    const expected = try constant(i8, &arena.allocator, .{
        .{
            .{ -1, 2 },
            .{ -3, 4 },
        },
        .{
            .{ -5, 6 },
            .{ -7, 8 },
        },
    });
    expectEqual(i8, actual, expected);
}

pub fn negateBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
    std.debug.assert(context.forward_inputs.len == 1);
    const input = context.forward_inputs[0];
    const outputs = try context.allocator.alloc(CpuTensor(T), 1);
    errdefer context.allocator.free(outputs);

    switch (context.gradient_input.storage) {
        .scalar => |scalar| {
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .scalar = -scalar },
            };
        },
        .array => |array| {
            const input_array = input.storage.array;
            var new_array = try context.allocator.alloc(T, input_array.len);
            for (new_array) |*e, i| e.* = -array[i];
            outputs[0] = CpuTensor(T){
                .shape = input.shape,
                .stride = input.stride,
                .storage = .{ .array = new_array },
            };
        },
    }
    return outputs;
}

test "negate backward rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, -4);
    const gradient_input = try constant(f64, &arena.allocator, 1);
    const actual = try negateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(f64, &arena.allocator, -1);
    expectEqual(f64, actual[0], expected);
}

test "negate backward rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{ 0, 2, -3, 4, -5 });
    const gradient_input = try constant(f64, &arena.allocator, .{ 2, 4, 6, 8, 10 });
    const actual = try negateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(f64, &arena.allocator, .{ -2, -4, -6, -8, -10 });
    expectEqual(f64, actual[0], expected);
}

test "negate backward rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(f64, &arena.allocator, .{
        .{ 0, -2 },
        .{ 3, -4 },
    });
    const gradient_input = try constant(f64, &arena.allocator, .{
        .{ 2, 4 },
        .{ 6, 8 },
    });
    const actual = try negateBackward(f64, backward.Context(f64){
        .allocator = &arena.allocator,
        .gradient_input = gradient_input,
        .forward_inputs = &[_]CpuTensor(f64){x},
    });
    const expected = try constant(f64, &arena.allocator, .{
        .{ -2, -4 },
        .{ -6, -8 },
    });
    expectEqual(f64, actual[0], expected);
}
