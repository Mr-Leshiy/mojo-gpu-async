from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.testing import TestSuite, assert_equal

from gpu_async.context import Context
from gpu_async.executor import Executor


def square_kernel(buf: Pointer[Float32, MutAnyOrigin]):
    var idx = global_idx.x

    var value = buf[unsafe_offset=idx]
    buf[unsafe_offset=idx] = value * value


async def square[
    size: Int
](ctx: Context, input: Array[Float32, size]) raises -> Array[Float32, size]:
    var device_buffer = ctx.gpu_ctx().enqueue_create_buffer[DType.float32](size)
    ctx.gpu_ctx().enqueue_copy(
        dst_buf=device_buffer, src_ptr=input.unsafe_ptr()
    )
    await ctx.synchronize()

    ctx.gpu_ctx().enqueue_function[square_kernel](
        device_buffer, grid_dim=1, block_dim=size
    )

    var result = Array[Float32, size](uninitialized=True)
    ctx.gpu_ctx().enqueue_copy(
        dst_ptr=result.unsafe_ptr(), src_buf=device_buffer
    )
    await ctx.synchronize()
    return result^


def test_square_kernel_runs_through_executor() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)
        var context = executor.context()

        comptime SIZE = 8

        var input1: Array[Float32, SIZE] = [1, 2, 3, 4, 5, 6, 7, 8]
        var t1 = executor.add(square(context, input1))

        var input2: Array[Float32, SIZE] = [8, 7, 6, 5, 4, 3, 2, 1]
        var t2 = executor.add(square(context, input2))

        var input3: Array[Float32, SIZE] = [4, 3, 2, 1, 8, 7, 6, 5]
        var t3 = executor.add(square(context, input3))

        assert_equal(t3^.wait(), [16, 9, 4, 1, 64, 49, 36, 25])
        assert_equal(t2^.wait(), [64, 49, 36, 25, 16, 9, 4, 1])
        assert_equal(t1^.wait(), [1, 4, 9, 16, 25, 36, 49, 64])


def main() raises:
    # TODO: https://github.com/modular/modular/issues/6890
    # TestSuite.discover_tests[__functions_in_module()]().run()
    test_square_kernel_runs_through_executor()
