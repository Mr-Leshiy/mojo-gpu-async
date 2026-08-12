"""Exercises the executor against a real GPU kernel, not just `synchronize()`.

Squares every element of a buffer on the device, yielding back to the
executor between the kernel launch and the copy-back so the sync it needs
goes through the same lazy, once-per-batch path `Context.synchronize()`
always takes.
"""

from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal

from gpu_async.context import Context
from gpu_async.executor import Executor


def _square_kernel(buf: Pointer[Float32, MutAnyOrigin], size: Int):
    """Square each element of `buf` in place, one thread per element."""
    var idx = global_idx.x
    if idx < size:
        var value = buf[unsafe_offset=idx]
        buf[unsafe_offset=idx] = value * value


async def _square_on_gpu[
    size: Int
](context: Context, input: Array[Float32, size]) raises -> Array[Float32, size]:
    var device_buffer = context.gpu_ctx().enqueue_create_buffer[DType.float32](
        size
    )
    context.gpu_ctx().enqueue_copy(
        dst_buf=device_buffer, src_ptr=input.unsafe_ptr()
    )
    await context.synchronize()

    context.gpu_ctx().enqueue_function[_square_kernel](
        device_buffer, size, grid_dim=1, block_dim=size
    )

    # The kernel launch above only queues work on the device; hand control
    # back to the executor rather than blocking here. It synchronizes the
    # device lazily, right before resuming this coroutine.
    await context.synchronize()

    var result = Array[Float32, size](uninitialized=True)
    context.gpu_ctx().enqueue_copy(
        dst_ptr=result.unsafe_ptr(), src_buf=device_buffer
    )
    await context.synchronize()

    return result^


def test_square_kernel_runs_through_executor() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            comptime SIZE = 8
            var input: Array[Float32, SIZE] = [1, 2, 3, 4, 5, 6, 7, 8]

            var task = executor.add(_square_on_gpu[SIZE](context, input))
            executor.wait()

            var result = task^.wait()
            assert_equal(result, [1, 4, 9, 16, 25, 36, 49, 64])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
