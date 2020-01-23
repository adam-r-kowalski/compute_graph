#!/bin/bash
zig test src/main.zig --cache off --verbose-link
lldb test
