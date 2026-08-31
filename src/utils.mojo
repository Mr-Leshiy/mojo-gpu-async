"""Small debugging helpers."""

from std.builtin._coroutine import AnyCoroutine


def _coroutine_address(hdl: AnyCoroutine) -> String:
    """Print the address of a coroutine handle.

    `AnyCoroutine` is an opaque MLIR type, not a pointer, so there's no
    direct way to read its bits. Taking a `Pointer` to the local `hdl` and
    reinterpreting *that* as a `Pointer[Int]` gives a view onto the same
    (pointer-sized) bytes, which can then be read as an address.

    Args:
        hdl: The coroutine handle to print the address of.
    """
    return hex(UInt64(Pointer(to=hdl).unsafe_bitcast[Int]()[]))
