test "" {
    _ = @import("lazy.zig");
    _ = @import("eager.zig");
    _ = @import("util/array_info.zig");
}
