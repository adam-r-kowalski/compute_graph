pub const Operation = @import("lazy/operation.zig").Operation;
pub const Graph = @import("lazy/graph.zig").Graph;
pub const absolute = @import("lazy/absolute.zig").absolute;
pub const add = @import("lazy/add.zig").add;
pub const constant = @import("lazy/constant.zig").constant;
pub const multiply = @import("lazy/multiply.zig").multiply;
pub const Node = @import("lazy/node.zig").Node;
pub const Session = @import("lazy/session.zig").Session;
pub const subtract = @import("lazy/subtract.zig").subtract;
pub const Tensor = @import("lazy/tensor.zig").Tensor;

test "" {
    const std = @import("std");
    std.meta.refAllDecls(@This());
}
