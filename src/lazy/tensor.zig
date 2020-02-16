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
};
