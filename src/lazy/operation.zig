const Allocator = @import("std").mem.Allocator;
const CpuTensorUnion = @import("../eager.zig").CpuTensorUnion;
const Tensor = @import("tensor.zig").Tensor;

pub const Operation = struct {
    pub const ForwardError = error{
        OutOfMemory,
        ShapeMismatch,
        CouldNotBroadcastShapes,
        Overflow,
        OperationNotDefinedForScalarType,
        InvalidDimension,
    };

    pub const BackwardError = error{
        OutOfMemory,
        ShapeMismatch,
        CouldNotBroadcastShapes,
        Overflow,
        OperationNotDefinedForScalarType,
        InvalidDimension,
        CannotDifferentiateIntegral,
    };

    pub const ForwardContext = struct {
        operation: *const Operation,
        allocator: *Allocator,
        values: []const CpuTensorUnion,
    };

    pub const BackwardContext = struct {
        operation: *const Operation,
        allocator: *Allocator,
        gradient_input: CpuTensorUnion,
        forward_inputs: []const CpuTensorUnion,
    };

    pub const ForwardResult = ForwardError!CpuTensorUnion;
    pub const BackwardResult = BackwardError![]const CpuTensorUnion;

    inputs: fn (self: *const Operation) []const Tensor,
    forward: fn (context: ForwardContext) ForwardResult,
    backward: ?fn (context: BackwardContext) BackwardResult,
};
