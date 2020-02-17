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
    f64,
    f32,
    f16,
    i64,
    i32,
    i8
};

pub const Tensor = struct {
    tensorType: TensorType,
    shape: []const usize,
    scalarType: ScalarType,

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        context: var,
        comptime Errors: type,
        comptime output: fn (@TypeOf(context), []const u8) Errors!void,
    ) Errors!void {
        try std.fmt.format(context, Errors, output, "Tensor(", .{});
        var i: usize = 0;
        const len = self.shape.len;
        while (i < len) : (i += 1)
            try std.fmt.format(context, Errors, output, "[{}]", .{self.shape[i]});
        switch(self.scalarType) {
            .f64 => try std.fmt.format(context, Errors, output, "f64", .{}),
            .f32 => try std.fmt.format(context, Errors, output, "f32", .{}),
            .f16 => try std.fmt.format(context, Errors, output, "f16", .{}),
            .i64 => try std.fmt.format(context, Errors, output, "i64", .{}),
            .i32 => try std.fmt.format(context, Errors, output, "i32", .{}),
            .i8 => try std.fmt.format(context, Errors, output, "i8", .{}),
        }
        try std.fmt.format(context, Errors, output, ")", .{});
    }
};
