pub const Operation = @import("lazy/operation.zig").Operation;
pub const Graph = @import("lazy/graph.zig").Graph;
pub const absolute = @import("lazy/absolute.zig").absolute;
pub const add = @import("lazy/add.zig").add;
pub const constant = @import("lazy/constant.zig").constant;
// pub const gradient = @import("lazy/gradient.zig").gradient;
pub const multiply = @import("lazy/multiply.zig").multiply;
pub const matrix_multiply = @import("lazy/matrix_multiply.zig").matrix_multiply;
pub const mean = @import("lazy/mean.zig").mean;
pub const Node = @import("lazy/node.zig").Node;
pub const Session = @import("lazy/session.zig").Session;
pub const subtract = @import("lazy/subtract.zig").subtract;
pub const Tensor = @import("lazy/tensor.zig").Tensor;

test "" {
    const std = @import("std");
    std.meta.refAllDecls(@This());
}
