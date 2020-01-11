test "" {
    _ = @import("absolute.zig");
    _ = @import("add.zig");
    _ = @import("constant.zig");
    _ = @import("graph.zig");
    _ = @import("multiply.zig");
    _ = @import("matrix_multiply.zig");
    _ = @import("node.zig");
    _ = @import("operation.zig");
    _ = @import("session.zig");
    _ = @import("subtract.zig");
    _ = @import("tensor.zig");

    _ = @import("eager/backup.zig");
    _ = @import("eager/constant.zig");
    _ = @import("eager/absolute.zig");

    _ = @import("util/array_info.zig");
}
