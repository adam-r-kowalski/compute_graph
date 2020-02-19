// TODO(Adam): Clean up circular dependency
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;

pub const Variable = struct {
    current_value: Tensor,
};

pub fn variable(graph: *Graph, initial_value: Tensor) !Tensor {
    try graph.variables.append(Variable{ .current_value = initial_value });
    return Tensor{
        .tensorType = .{ .variable = graph.variables.len - 1 },
        .shape = initial_value.shape,
        .scalarType = initial_value.scalarType,
    };
}
