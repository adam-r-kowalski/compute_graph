pub const Node = union(enum) {
    constant: u64,
    operation: u64,
};
