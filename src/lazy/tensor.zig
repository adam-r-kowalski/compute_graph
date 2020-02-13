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

pub const Tensor = struct {
    tensorType: TensorType,
};
