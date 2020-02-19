const std = @import("std");
const Allocator = std.mem.Allocator;
const constant = @import("constant.zig").constant;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const CpuStorage = cpu_tensor.CpuStorage;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;
const expectEqual = @import("../testing.zig").expectEqual;
const backward = @import("backward.zig");

pub fn sum(comptime T: type, allocator: *Allocator, tensor: CpuTensor(T)) !CpuTensor(T) {
    const shape = try allocator.alloc(usize, 0);
    errdefer allocator.free(shape);
    const stride = try allocator.alloc(usize, 0);
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
    const actual = try sum(f64, &arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f64, -5));
    expectEqual(f64, actual, expected);
}

test "sum rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try constant(&arena.allocator, [_]i32{ 5, 10, 7, 8, 10 });
    const actual = try sum(i32, &arena.allocator, x);
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
    const actual = try sum(f16, &arena.allocator, x);
    const expected = try constant(&arena.allocator, @as(f16, 48));
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
    const actual = try sum(i8, &arena.allocator, x);
    std.debug.warn("\n{}\n", .{actual});
    const expected = try constant(&arena.allocator, @as(i8, 56));
    expectEqual(i8, actual, expected);
}


// pub fn sumBackward(comptime T: type, context: backward.Context(T)) ![]CpuTensor(T) {
//     std.debug.assert(context.forward_inputs.len == 1);
//     const input = context.forward_inputs[0];
//     const outputs = try context.allocator.alloc(CpuTensor(T), 1);
//     errdefer context.allocator.free(outputs);
//     const scalar = context.gradient_input.storage.scalar;
//     const value = scalar / @intToFloat(T, length(T, input));
//     outputs[0] = try fill(T, context.allocator, value, context.forward_inputs[0].shape);
//     return outputs;
// }

// test "sum backward rank 0" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const forward_input = try constant(&arena.allocator, @as(f64, 4));
//     const gradient_input = try constant(&arena.allocator, @as(f64, 1));
//     const actual = try sumBackward(f64, backward.Context(f64){
//         .allocator = &arena.allocator,
//         .gradient_input = gradient_input,
//         .forward_inputs = &[_]CpuTensor(f64){forward_input},
//     });
//     const expected = try constant(&arena.allocator, @as(f64, 1));
//     expectEqual(f64, actual[0], expected);
// }

// test "sum backward rank 1" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const forward_input = try constant(&arena.allocator, [_]f64{ 1, 2, 3, 4, 5 });
//     const gradient_input = try constant(&arena.allocator, @as(f64, 1));
//     const actual = try sumBackward(f64, backward.Context(f64){
//         .allocator = &arena.allocator,
//         .gradient_input = gradient_input,
//         .forward_inputs = &[_]CpuTensor(f64){forward_input},
//     });
//     const expected = try constant(&arena.allocator, [_]f64{ 0.2, 0.2, 0.2, 0.2, 0.2 });
//     expectEqual(f64, actual[0], expected);
// }

// test "sum backward rank 2" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const forward_input = try constant(&arena.allocator, [_][2]f64{
//         .{ 1, 2 },
//         .{ 3, 4 },
//     });
//     const gradient_input = try constant(&arena.allocator, @as(f64, 1));
//     const actual = try sumBackward(f64, backward.Context(f64){
//         .allocator = &arena.allocator,
//         .gradient_input = gradient_input,
//         .forward_inputs = &[_]CpuTensor(f64){forward_input},
//     });
//     const expected = try constant(&arena.allocator, [_][2]f64{
//         .{ 0.25, 0.25 },
//         .{ 0.25, 0.25 },
//     });
//     expectEqual(f64, actual[0], expected);
// }
