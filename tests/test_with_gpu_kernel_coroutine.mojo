from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import global_idx
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal

from warp.context import Context
from warp.executor import Executor


# Workaround for a nightly compiler bug: helpers keep GPU host-API calls out of the `async def` body and the kernel out of module scope.
def _copy_to_device[
    size: Int
](
    ctx: Context,
    device_buffer: DeviceBuffer[DType.float32],
    src: Pointer[Float32, ImmutAnyOrigin],
) raises:
    ctx.gpu_ctx().enqueue_copy(dst_buf=device_buffer, src_ptr=src)


def _launch_square_kernel[
    size: Int
](ctx: Context, device_buffer: DeviceBuffer[DType.float32]) raises:
    def square_kernel(buf: Pointer[Float32, MutAnyOrigin]):
        var idx = global_idx.x

        var value = buf[unsafe_offset=idx]
        buf[unsafe_offset=idx] = value * value

    ctx.gpu_ctx().enqueue_function[square_kernel](
        device_buffer, grid_dim=1, block_dim=size
    )


def _copy_from_device[
    size: Int
](
    ctx: Context,
    dst: Pointer[Float32, MutAnyOrigin],
    device_buffer: DeviceBuffer[DType.float32],
) raises:
    ctx.gpu_ctx().enqueue_copy(dst_ptr=dst, src_buf=device_buffer)


async def square[
    size: Int
](ctx: Context, input: Array[Float32, size]) raises -> Array[Float32, size]:
    var device_buffer = ctx.gpu_ctx().enqueue_create_buffer[DType.float32](size)
    _copy_to_device[size](
        ctx,
        device_buffer,
        input.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
    )
    await ctx.synchronize()

    _launch_square_kernel[size](ctx, device_buffer)

    var result = Array[Float32, size](uninitialized=True)
    _copy_from_device[size](
        ctx,
        result.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        device_buffer,
    )
    await ctx.synchronize()
    return result^


def test_square_kernel_runs_through_executor() raises:
    comptime if has_accelerator():
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
    TestSuite.discover_tests[__functions_in_module()]().run()
