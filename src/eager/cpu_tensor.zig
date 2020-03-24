const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const tensorScalarType = @import("../lazy/tensor.zig").ScalarType;

pub fn CpuStorage(comptime ScalarType: type) type {
    return union(enum) {
        scalar: ScalarType,
        array: []const ScalarType,
    };
}

fn printIndent(out_stream: var, depth: usize) !void {
    var i: usize = 0;
    while (i < depth * 2) : (i += 1)
        try std.fmt.format(out_stream, " ", .{});
}

fn printTensor(
    comptime T: type,
    out_stream: var,
    depth: usize,
    shape: []const usize,
    stride: []const usize,
    array: []const T,
    comma: bool,
) @TypeOf(out_stream).Error!void {
    if (shape.len == 0)
        return;
    try printIndent(out_stream, depth);
    if (depth != 0)
        try std.fmt.format(out_stream, ".", .{});
    try std.fmt.format(out_stream, "{{", .{});

    if (shape.len == 1) {
        const len = shape[0];
        var i: usize = 0;
        try std.fmt.format(out_stream, " ", .{});
        while (i < len) : (i += 1) {
            try std.fmt.format(out_stream, "{}", .{array[i]});
            if (i < len - 1)
                try std.fmt.format(out_stream, ", ", .{});
        }
        try std.fmt.format(out_stream, " ", .{});
    } else {
        try std.fmt.format(out_stream, "\n", .{});
        const len = shape[0];
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const start = i * stride[0];
            const end = start + stride[0];
            try printTensor(T, out_stream, depth + 1, shape[1..], stride[1..], array[start..end], i < len - 1);
        }
    }
    if (shape.len > 1)
        try printIndent(out_stream, depth);
    try std.fmt.format(out_stream, "}}", .{});
    if (depth != 0) {
        if (comma)
            try std.fmt.format(out_stream, ",", .{});
        try std.fmt.format(out_stream, "\n", .{});
    }
}

pub fn CpuTensor(comptime T: type) type {
    return struct {
        shape: []const usize,
        stride: []const usize,
        storage: CpuStorage(T),

        pub const ScalarType = T;

        pub fn deinit(self: @This(), allocator: *Allocator) void {
            allocator.free(self.shape);
            allocator.free(self.stride);
            switch (self.storage) {
                .array => |array| allocator.free(array),
                .scalar => {},
            }
        }

        fn copy(self: @This(), allocator: *Allocator) !@This() {
            const shape = try allocator.alloc(usize, self.shape.len);
            errdefer allocator.free(shape);
            for (self.shape) |s, i| shape[i] = s;
            const stride = try allocator.alloc(usize, self.stride.len);
            errdefer allocator.free(stride);
            for (self.stride) |s, i| stride[i] = s;
            switch (self.storage) {
                .scalar => |scalar| {
                    return CpuTensor(T){
                        .shape = shape,
                        .stride = stride,
                        .storage = .{ .scalar = scalar },
                    };
                },
                .array => |array| {
                    const new_array = try allocator.alloc(T, array.len);
                    errdefer allocator.free(new_array);
                    for (array) |a, i| new_array[i] = a;
                    return CpuTensor(T){
                        .shape = shape,
                        .stride = stride,
                        .storage = .{ .array = new_array },
                    };
                },
            }
        }

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            out_stream: var,
        ) !void {
            try std.fmt.format(out_stream, "CpuTensor(", .{});
            switch (self.storage) {
                .scalar => |scalar| {
                    try std.fmt.format(out_stream, "@as({}, {})", .{ @typeName(T), scalar });
                },
                .array => |array| {
                    var i: usize = 0;
                    const len = self.shape.len;
                    while (i < len) : (i += 1)
                        try std.fmt.format(out_stream, "[{}]", .{self.shape[i]});
                    try std.fmt.format(out_stream, "{}", .{@typeName(T)});
                    try printTensor(T, out_stream, 0, self.shape, self.stride, array, true);
                },
            }
            try std.fmt.format(out_stream, ")", .{});
        }
    };
}

pub const CpuTensorUnion = union(tensorScalarType) {
    f64: CpuTensor(f64),
    f32: CpuTensor(f32),
    f16: CpuTensor(f16),
    i64: CpuTensor(i64),
    i32: CpuTensor(i32),
    i8: CpuTensor(i8),

    pub fn init(tensor: var) CpuTensorUnion {
        return switch (@TypeOf(tensor).ScalarType) {
            f64 => .{ .f64 = tensor },
            f32 => .{ .f32 = tensor },
            f16 => .{ .f16 = tensor },
            i64 => .{ .i64 = tensor },
            i32 => .{ .i32 = tensor },
            i8 => .{ .i8 = tensor },
            else => @compileError("ScalarType not supported"),
        };
    }

    pub fn deinit(self: @This(), allocator: *Allocator) void {
        switch (self) {
            .f64 => |tensor| tensor.deinit(allocator),
            .f32 => |tensor| tensor.deinit(allocator),
            .f16 => |tensor| tensor.deinit(allocator),
            .i64 => |tensor| tensor.deinit(allocator),
            .i32 => |tensor| tensor.deinit(allocator),
            .i8 => |tensor| tensor.deinit(allocator),
        }
    }

    fn copy(self: @This(), allocator: *Allocator) !@This() {
        return switch (self) {
            .f64 => |tensor| .{ .f64 = try tensor.copy(allocator) },
            .f32 => |tensor| .{ .f32 = try tensor.copy(allocator) },
            .f16 => |tensor| .{ .f16 = try tensor.copy(allocator) },
            .i64 => |tensor| .{ .i64 = try tensor.copy(allocator) },
            .i32 => |tensor| .{ .i32 = try tensor.copy(allocator) },
            .i8 => |tensor| .{ .i8 = try tensor.copy(allocator) },
        };
    }

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        out_stream: var,
    ) !void {
        switch (self) {
            .f64 => |tensor| try std.fmt.format(out_stream, "{}", .{tensor}),
            .f32 => |tensor| try std.fmt.format(out_stream, "{}", .{tensor}),
            .f16 => |tensor| try std.fmt.format(out_stream, "{}", .{tensor}),
            .i64 => |tensor| try std.fmt.format(out_stream, "{}", .{tensor}),
            .i32 => |tensor| try std.fmt.format(out_stream, "{}", .{tensor}),
            .i8 => |tensor| try std.fmt.format(out_stream, "{}", .{tensor}),
        }
    }
};

pub fn tensorStride(allocator: *Allocator, shape: []const usize) ![]usize {
    const rank = shape.len;
    var stride = try allocator.alloc(usize, rank);
    errdefer allocator.free(stride);
    if (rank == 0)
        return stride;
    stride[rank - 1] = 1;
    var i: usize = rank - 1;
    var product: usize = 1;
    while (i > 0) : (i -= 1) {
        product *= shape[i];
        stride[i - 1] = product;
    }
    return stride;
}

test "stride rank 0" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{};
    const stride = try tensorStride(allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{}));
}

test "stride rank 3" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 2, 3 };
    const stride = try tensorStride(allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{ 6, 3, 1 }));
}

test "stride rank 4" {
    const allocator = std.heap.page_allocator;
    const shape = [_]usize{ 3, 4, 5, 6 };
    const stride = try tensorStride(allocator, shape[0..]);
    defer allocator.free(stride);
    expect(std.mem.eql(usize, stride, &[_]usize{ 120, 30, 6, 1 }));
}

pub fn tensorLength(shape: []const usize) usize {
    var length: usize = 1;
    for (shape) |s| length *= s;
    return length;
}

test "length rank 0" {
    const shape = [_]usize{};
    expectEqual(tensorLength(shape[0..]), 1);
}

test "length rank 3" {
    const shape = [_]usize{ 3, 2, 3 };
    expectEqual(tensorLength(shape[0..]), 18);
}

test "length rank 4" {
    const shape = [_]usize{ 3, 4, 5, 6 };
    expectEqual(tensorLength(shape[0..]), 360);
}

pub fn linearIndex(stride: []const usize, cartesian_index: []const usize) usize {
    var index: usize = 0;
    for (stride) |s, i| index += s * cartesian_index[i];
    return index;
}

test "linear index" {
    const stride = &[_]usize{ 3, 1 };
    std.testing.expectEqual(linearIndex(stride, &[_]usize{ 0, 0 }), 0);
    std.testing.expectEqual(linearIndex(stride, &[_]usize{ 0, 1 }), 1);
    std.testing.expectEqual(linearIndex(stride, &[_]usize{ 0, 2 }), 2);
    std.testing.expectEqual(linearIndex(stride, &[_]usize{ 1, 0 }), 3);
    std.testing.expectEqual(linearIndex(stride, &[_]usize{ 1, 1 }), 4);
    std.testing.expectEqual(linearIndex(stride, &[_]usize{ 1, 2 }), 5);
}
