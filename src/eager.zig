pub const absolute = @import("eager/absolute.zig").absolute;
pub const add = @import("eager/add.zig").add;
pub const constant = @import("eager/constant.zig").constant;
pub const exponentiate = @import("eager/exponentiate.zig").exponentiate;
pub const multiply = @import("eager/multiply.zig").multiply;
pub const matrixMultiply = @import("eager/matrix_multiply.zig").matrixMultiply;
pub const mean = @import("eager/mean.zig").mean;
pub const negate = @import("eager/negate.zig").negate;
pub const subtract = @import("eager/subtract.zig").subtract;
pub const onesLike = @import("eager/ones_like.zig").onesLike;
const cpu_tensor = @import("eager/cpu_tensor.zig");
pub const CpuTensor = cpu_tensor.CpuTensor;
pub const CpuTensorUnion = cpu_tensor.CpuTensorUnion;

test "" {
    const std = @import("std");
    std.meta.refAllDecls(@This());
}
