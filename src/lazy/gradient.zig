const std = @import("std");
// TODO(Adam): Clean up circular dependency
const Graph = @import("graph.zig").Graph;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const GradientHandle = tensor.GradientHandle;

pub const Gradient = struct {
    of: Tensor,
    with_respect_to: []const Tensor,
};

pub fn gradient(graph: *Graph, of: Tensor, with_respect_to: []const Tensor) ![]Tensor {
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
        .shape = &[_]usize{},
    };
    return gradients;
}
