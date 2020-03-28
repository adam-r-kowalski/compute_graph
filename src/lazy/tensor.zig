const std = @import("std");

pub const GradientHandle = struct {
    gradient: usize,
    index: usize,
};

const TensorType = union(enum) {
    constant: usize,
    operation: usize,
    gradient_handle: GradientHandle,
    variable: usize,
    assign: usize,
    placeholder: usize,
};

pub const ScalarType = enum {
    f64, f32, f16, i64, i32, i8
};

/// Tensors are a generalization of arrays to N dimensions.
///
/// For example:
/// Scalars are 0 dimensional tensors.
/// Vectors are 1 dimensional tensors.
/// Matrices are 2 dimensional tensors.
///
/// Tensors allow you to deal with data of arbitrary dimensionality with a uniform abstraction.
/// Operations on tensors are lazy and will build a computation graph which can be manipulated
/// and executed by sessions.
pub const Tensor = struct {
    tensorType: TensorType,
    shape: []const usize,
    scalarType: ScalarType,

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        out_stream: var,
    ) !void {
        try std.fmt.format(out_stream, "Tensor(", .{});
        var i: usize = 0;
        const len = self.shape.len;
        while (i < len) : (i += 1)
            try std.fmt.format(out_stream, "[{}]", .{self.shape[i]});
        switch (self.scalarType) {
            .f64 => try std.fmt.format(out_stream, "f64", .{}),
            .f32 => try std.fmt.format(out_stream, "f32", .{}),
            .f16 => try std.fmt.format(out_stream, "f16", .{}),
            .i64 => try std.fmt.format(out_stream, "i64", .{}),
            .i32 => try std.fmt.format(out_stream, "i32", .{}),
            .i8 => try std.fmt.format(out_stream, "i8", .{}),
        }
        try std.fmt.format(out_stream, ")", .{});
    }
};
