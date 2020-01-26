const std = @import("std");
const Graph = @import("graph.zig").Graph;
const Tensor = @import("tensor.zig").Tensor;

pub const Gradient = struct {
    of: Tensor,
    with_respect_to: []const Tensor,
};

pub fn gradient(graph: *Graph, of: Tensor, with_respect_to: []const Tensor) ![]Tensor {
    try graph.gradients.append(.{
        .of = of,
        .with_respect_to = with_respect_to,
    });
    const gradients = try graph.arena.allocator.alloc(Tensor, with_respect_to.len);
    var i: usize = 0;
    while (i < gradients.len) : (i += 1)
        gradients[i] = Tensor{ .gradient = graph.gradients.len - 1 };
    // TODO(Adam) associate index with each gradient
    return gradients;
}
