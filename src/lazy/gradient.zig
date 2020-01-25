const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;

pub const Gradient = struct {
    of: Tensor,
    with_respect_to: Tensor,
};

pub fn gradient(graph: *Graph, of: Tensor, with_respect_to: Tensor) !Tensor {
    try graph.gradients.append(.{
        .of = of,
        .with_respect_to = with_respect_to,
    });
    return Tensor{ .gradient = graph.gradients.len - 1 };
}
