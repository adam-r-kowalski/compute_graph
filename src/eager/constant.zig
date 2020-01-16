const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const arrayInfo = @import("../util/array_info.zig").arrayInfo;
const cpu_tensor = @import("cpu_tensor.zig");
const CpuTensor = cpu_tensor.CpuTensor;
const CpuStorage = cpu_tensor.CpuStorage;
const tensorStride = cpu_tensor.tensorStride;
const tensorLength = cpu_tensor.tensorLength;

fn ConstantType(comptime T: type) type {
    return CpuTensor(arrayInfo(T).ScalarType);
}

fn transferMemory(comptime ScalarType: type, array: []ScalarType, literal: var, index: *usize) void {
    switch (@typeInfo(@TypeOf(literal))) {
        .Pointer, .Array => {
            var i: usize = 0;
            while (i < literal.len) : (i += 1)
                transferMemory(ScalarType, array, literal[i], index);
        },
        else => {
            array[index.*] = literal;
            index.* += @as(usize, 1);
        },
    }
}

fn tensorShape(comptime rank: usize, allocator: *Allocator, literal: var) ![]usize {
    var shape = try allocator.alloc(usize, rank);
    errdefer allocator.free(shape);
    const Closure = struct {
        fn call(s: []usize, i: usize, l: var) void {
            switch (@typeInfo(@TypeOf(l))) {
                .Pointer, .Array => {
                    s[i] = l.len;
                    call(s, i + 1, l[0]);
                },
                else => {},
            }
        }
    };
    Closure.call(shape, 0, literal);
    return shape;
}

pub fn constant(allocator: *Allocator, literal: var) !ConstantType(@TypeOf(literal)) {
    const info = arrayInfo(@TypeOf(literal));
    const T = CpuTensor(info.ScalarType);
    const shape = try tensorShape(info.rank, allocator, literal);
    const stride = try tensorStride(info.rank, allocator, shape);
    if (info.rank == 0) {
        return T{
            .shape = shape,
            .stride = stride,
            .storage = CpuStorage(info.ScalarType){ .scalar = literal },
        };
    }
    var array = try allocator.alloc(info.ScalarType, tensorLength(shape));
    errdefer allocator.free(array);
    var index: usize = 0;
    transferMemory(info.ScalarType, array, literal, &index);
    return T{
        .shape = shape,
        .stride = stride,
        .storage = CpuStorage(info.ScalarType){ .array = array },
    };
}

test "constant rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, @as(f16, 5));
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{}));
    expectEqual(tensor.storage.scalar, 5);
}

test "constant rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, &[_]f64{ 1, 2, 3 });
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{3}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{1}));
    expect(std.mem.eql(f64, tensor.storage.array, &[_]f64{ 1, 2, 3 }));
}

test "constant rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, &[_][3]i32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 2, 3 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 3, 1 }));
    expect(std.mem.eql(i32, tensor.storage.array, &[_]i32{ 1, 2, 3, 4, 5, 6 }));
}

test "constant rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(&arena.allocator, &[_][2][3]f16{
        .{
            .{ 1, 2, 3 },
            .{ 4, 5, 6 },
        },
        .{
            .{ 7, 8, 9 },
            .{ 10, 11, 12 },
        },
        .{
            .{ 13, 14, 15 },
            .{ 16, 17, 18 },
        },
    });
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 3, 2, 3 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 6, 3, 1 }));
    expect(std.mem.eql(f16, tensor.storage.array, &[_]f16{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    }));
}
