const std = @import("std");

fn ReturnType(comptime Invokable: type) type {
    switch (@typeInfo(Invokable)) {
        .Fn => |f| {
            return f.return_type.?;
        },
        .Struct => {
            return @typeInfo(@TypeOf(Invokable.call)).Fn.return_type.?;
        },
        else => @compileError("not an invokable object"),
    }
}

fn invoke(invokable: var, args: var) ReturnType(@TypeOf(invokable)) {
    switch (@typeInfo(@TypeOf(invokable))) {
        .Fn => return @call(.{}, invokable, args),
        .Struct => return @call(.{}, invokable.call, args),
        else => @compileError("not an invokable object"),
    }
}

test "invoke function pointer" {
    const callable = struct {
        fn call(x: i32) i32 {
            return x + 2;
        }
    };
    std.testing.expectEqual(invoke(callable.call, .{10}), 12);
}

test "invoke closure" {
    const Callable = struct {
        x: i32,

        fn call(self: @This(), y: i32) i32 {
            return self.x + y;
        }
    };
    const callable = Callable{ .x = 5 };
    std.testing.expectEqual(invoke(callable, .{10}), 15);
}
