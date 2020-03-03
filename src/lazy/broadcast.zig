const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const eager = @import("../eager.zig");

pub fn broadcastShape(allocator: *Allocator, x: Tensor, y: Tensor) ![]const usize {
    if (std.mem.eql(usize, x.shape, y.shape))
        return x.shape;
    if (x.shape.len == 0)
        return y.shape;
    if (y.shape.len == 0)
        return x.shape;
    return try eager.broadcast.broadcastShape(allocator, x.shape, y.shape);
}
