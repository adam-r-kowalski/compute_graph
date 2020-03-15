const std = @import("std");
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const GradientHandle = tensor.GradientHandle;

pub const Gradient = struct {
    of: Tensor,
    with_respect_to: []const Tensor,
};

pub fn gradient(graph: *Graph, of: Tensor, with_respect_to: []const Tensor) ![]Tensor {
    if (of.shape.len != 0) return error.requestingGradientOfNonScalar;
    try graph.gradients.append(Gradient{
        .of = of,
        .with_respect_to = with_respect_to,
    });
    const gradients = try graph.arena.allocator.alloc(Tensor, with_respect_to.len);
    var i: usize = 0;
    while (i < gradients.len) : (i += 1)
        gradients[i] = Tensor{
        .tensorType = .{
            .gradient_handle = GradientHandle{
                .gradient = graph.gradients.len - 1,
                .index = i,
            },
        },
        .shape = with_respect_to[i].shape,
        .scalarType = with_respect_to[i].scalarType,
    };
    return gradients;
}
