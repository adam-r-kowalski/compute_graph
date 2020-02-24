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

fn transferMemory(comptime T: type, array: []T, literal: var, index: *usize) void {
    switch (@typeInfo(@TypeOf(literal))) {
        .Pointer, .Array => {
            for (literal) |e|
                transferMemory(T, array, e, index);
        },
        .Struct => |s| {
            comptime var i: usize = 0;
            inline while (i < s.fields.len) : (i += 1)
                transferMemory(T, array, @field(literal, s.fields[i].name), index);
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
                .Struct => {
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

pub fn constant(comptime T: type, allocator: *Allocator, literal: var) !CpuTensor(T) {
    const info = arrayInfo(@TypeOf(literal));
    const shape = try tensorShape(info.rank, allocator, literal);
    errdefer allocator.free(shape);
    const stride = try tensorStride(allocator, shape);
    errdefer allocator.free(stride);
    if (info.rank == 0) {
        return CpuTensor(T){
            .shape = shape,
            .stride = stride,
            .storage = CpuStorage(T){ .scalar = @as(T, literal) },
        };
    }
    var array = try allocator.alloc(T, tensorLength(shape));
    errdefer allocator.free(array);
    var index: usize = 0;
    transferMemory(T, array, literal, &index);
    return CpuTensor(T){
        .shape = shape,
        .stride = stride,
        .storage = CpuStorage(T){ .array = array },
    };
}

test "constant rank 0" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(f16, &arena.allocator, 5);
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{}));
    expectEqual(tensor.storage.scalar, 5);
    const actual = try std.fmt.allocPrint(&arena.allocator, "{}", .{tensor});
    expect(std.mem.eql(u8, actual, "CpuTensor(@as(f16, 5.0e+00))"));
}

test "constant rank 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(f64, &arena.allocator, .{ 1, 2, 3 });
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{3}));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{1}));
    expect(std.mem.eql(f64, tensor.storage.array, &[_]f64{ 1, 2, 3 }));
    const actual = try std.fmt.allocPrint(&arena.allocator, "{}", .{tensor});
    expect(std.mem.eql(u8, actual, "CpuTensor([3]f64{ 1.0e+00, 2.0e+00, 3.0e+00 })"));
}

test "constant rank 2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(i32, &arena.allocator, .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 2, 3 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 3, 1 }));
    expect(std.mem.eql(i32, tensor.storage.array, &[_]i32{ 1, 2, 3, 4, 5, 6 }));
    const actual = try std.fmt.allocPrint(&arena.allocator, "{}", .{tensor});
    expect(std.mem.eql(u8, actual,
        \\CpuTensor([2][3]i32{
        \\  .{ 1, 2, 3 },
        \\  .{ 4, 5, 6 }
        \\})
    ));
}

test "constant rank 3" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(f16, &arena.allocator, .{
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
    const actual = try std.fmt.allocPrint(&arena.allocator, "{}", .{tensor});
    expect(std.mem.eql(u8, actual,
        \\CpuTensor([3][2][3]f16{
        \\  .{
        \\    .{ 1.0e+00, 2.0e+00, 3.0e+00 },
        \\    .{ 4.0e+00, 5.0e+00, 6.0e+00 }
        \\  },
        \\  .{
        \\    .{ 7.0e+00, 8.0e+00, 9.0e+00 },
        \\    .{ 1.0e+01, 1.1e+01, 1.2e+01 }
        \\  },
        \\  .{
        \\    .{ 1.3e+01, 1.4e+01, 1.5e+01 },
        \\    .{ 1.6e+01, 1.7e+01, 1.8e+01 }
        \\  }
        \\})
    ));
}

test "constant rank 4" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(i32, &arena.allocator, .{
        .{
            .{
                .{ 1, 2, 3, 4 },
                .{ 5, 6, 7, 8 },
                .{ 9, 10, 11, 12 },
            },
            .{
                .{ 13, 14, 15, 16 },
                .{ 17, 18, 19, 20 },
                .{ 21, 22, 23, 24 },
            },
        },
        .{
            .{
                .{ 25, 26, 27, 28 },
                .{ 29, 30, 31, 32 },
                .{ 33, 34, 35, 36 },
            },
            .{
                .{ 37, 38, 39, 40 },
                .{ 41, 42, 43, 44 },
                .{ 45, 46, 47, 48 },
            },
        },
        .{
            .{
                .{ 49, 50, 51, 52 },
                .{ 53, 54, 55, 56 },
                .{ 57, 58, 59, 60 },
            },
            .{
                .{ 61, 62, 63, 64 },
                .{ 65, 66, 67, 68 },
                .{ 69, 70, 71, 72 },
            },
        },
    });
    expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 3, 2, 3, 4 }));
    expect(std.mem.eql(usize, tensor.stride, &[_]usize{ 24, 12, 4, 1 }));
    expect(std.mem.eql(i32, tensor.storage.array, &[_]i32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    }));
    const actual = try std.fmt.allocPrint(&arena.allocator, "{}", .{tensor});
    expect(std.mem.eql(u8, actual,
        \\CpuTensor([3][2][3][4]i32{
        \\  .{
        \\    .{
        \\      .{ 1, 2, 3, 4 },
        \\      .{ 5, 6, 7, 8 },
        \\      .{ 9, 10, 11, 12 }
        \\    },
        \\    .{
        \\      .{ 13, 14, 15, 16 },
        \\      .{ 17, 18, 19, 20 },
        \\      .{ 21, 22, 23, 24 }
        \\    }
        \\  },
        \\  .{
        \\    .{
        \\      .{ 25, 26, 27, 28 },
        \\      .{ 29, 30, 31, 32 },
        \\      .{ 33, 34, 35, 36 }
        \\    },
        \\    .{
        \\      .{ 37, 38, 39, 40 },
        \\      .{ 41, 42, 43, 44 },
        \\      .{ 45, 46, 47, 48 }
        \\    }
        \\  },
        \\  .{
        \\    .{
        \\      .{ 49, 50, 51, 52 },
        \\      .{ 53, 54, 55, 56 },
        \\      .{ 57, 58, 59, 60 }
        \\    },
        \\    .{
        \\      .{ 61, 62, 63, 64 },
        \\      .{ 65, 66, 67, 68 },
        \\      .{ 69, 70, 71, 72 }
        \\    }
        \\  }
        \\})
    ));
}

test "CpuTensorUnion formatted printing" {
    const CpuTensorUnion = cpu_tensor.CpuTensorUnion;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const tensor = try constant(f16, &arena.allocator, .{
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
    const tensor_union = CpuTensorUnion.init(tensor);
    const actual = try std.fmt.allocPrint(&arena.allocator, "{}", .{tensor_union});
    expect(std.mem.eql(u8, actual,
        \\CpuTensor([3][2][3]f16{
        \\  .{
        \\    .{ 1.0e+00, 2.0e+00, 3.0e+00 },
        \\    .{ 4.0e+00, 5.0e+00, 6.0e+00 }
        \\  },
        \\  .{
        \\    .{ 7.0e+00, 8.0e+00, 9.0e+00 },
        \\    .{ 1.0e+01, 1.1e+01, 1.2e+01 }
        \\  },
        \\  .{
        \\    .{ 1.3e+01, 1.4e+01, 1.5e+01 },
        \\    .{ 1.6e+01, 1.7e+01, 1.8e+01 }
        \\  }
        \\})
    ));
}
