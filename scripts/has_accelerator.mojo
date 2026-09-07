"""Prints `true` or `false` for whether this host has a usable accelerator.

Kept separate from the test suite so it can be queried without pulling in
any GPU kernel code: `tests/test_gpu_synchronize.mojo` defines a real
kernel, and compiling that requires the toolchain to resolve a concrete GPU
architecture even on a host that will never execute it (`comptime if
has_accelerator():` only skips *running* the guarded body -- the body still
gets compiled, and kernel codegen fails hard with "Unknown GPU architecture
detected" when there's no hardware to detect). `has_accelerator()` alone
has no such requirement, so this query compiles and runs fine everywhere.
"""

from std.sys import has_accelerator


def main():
    print("true" if has_accelerator() else "false")
