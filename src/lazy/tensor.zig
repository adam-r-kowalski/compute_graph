pub const GradientHandle = struct {
    gradient: usize,
    index: usize,
};

pub const Tensor = union(enum) {
    constant: usize,
    operation: usize,
    gradient_handle: GradientHandle,
    variable: usize,
    assign: usize,
    placeholder: usize,
};
