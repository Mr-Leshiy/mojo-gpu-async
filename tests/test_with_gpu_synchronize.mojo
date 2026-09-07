from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import global_idx
from std.testing import TestSuite, assert_equal

from warp.context import Context
from warp.executor import Executor


# `square` is a stand-in kernel: any real device work would do. Workaround
# for a nightly compiler bug: helpers keep GPU host-API calls out of the
# `async def` body and the kernel out of module scope.
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

    _launch_square_kernel[size](ctx, device_buffer)

    var result = Array[Float32, size](uninitialized=True)
    _copy_from_device[size](
        ctx,
        result.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        device_buffer,
    )
    await ctx.synchronize()
    return result^


def test_synchronize_completes_a_real_device_round_trip() raises:
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


comptime _SCALE_SIZE = 8


def test_synchronize_scales_to_many_concurrent_gpu_tasks() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)
        var context = executor.context()

        # `Task` is immovable (see `warp/task.mojo`): the coroutine
        # writes its result and completion flag through raw pointers
        # into the `Task` struct's own memory, so a `Task` can't be
        # moved into a `List` once created. Sixteen separately-named
        # locals is the straightforward way to hold that many of them
        # alive at once. All sixteen are queued -- none driven -- before
        # the single `executor.wait()` below, so they really do run
        # concurrently on one executor rather than one at a time.
        var input: Array[Float32, _SCALE_SIZE] = [3, 3, 3, 3, 3, 3, 3, 3]

        var t1 = executor.add(square[_SCALE_SIZE](context, input))
        var t2 = executor.add(square[_SCALE_SIZE](context, input))
        var t3 = executor.add(square[_SCALE_SIZE](context, input))
        var t4 = executor.add(square[_SCALE_SIZE](context, input))
        var t5 = executor.add(square[_SCALE_SIZE](context, input))
        var t6 = executor.add(square[_SCALE_SIZE](context, input))
        var t7 = executor.add(square[_SCALE_SIZE](context, input))
        var t8 = executor.add(square[_SCALE_SIZE](context, input))
        var t9 = executor.add(square[_SCALE_SIZE](context, input))
        var t10 = executor.add(square[_SCALE_SIZE](context, input))
        var t11 = executor.add(square[_SCALE_SIZE](context, input))
        var t12 = executor.add(square[_SCALE_SIZE](context, input))
        var t13 = executor.add(square[_SCALE_SIZE](context, input))
        var t14 = executor.add(square[_SCALE_SIZE](context, input))
        var t15 = executor.add(square[_SCALE_SIZE](context, input))
        var t16 = executor.add(square[_SCALE_SIZE](context, input))

        executor.wait()

        var expected: Array[Float32, _SCALE_SIZE] = [9, 9, 9, 9, 9, 9, 9, 9]
        assert_equal(t1^.wait(), expected)
        assert_equal(t2^.wait(), expected)
        assert_equal(t3^.wait(), expected)
        assert_equal(t4^.wait(), expected)
        assert_equal(t5^.wait(), expected)
        assert_equal(t6^.wait(), expected)
        assert_equal(t7^.wait(), expected)
        assert_equal(t8^.wait(), expected)
        assert_equal(t9^.wait(), expected)
        assert_equal(t10^.wait(), expected)
        assert_equal(t11^.wait(), expected)
        assert_equal(t12^.wait(), expected)
        assert_equal(t13^.wait(), expected)
        assert_equal(t14^.wait(), expected)
        assert_equal(t15^.wait(), expected)
        assert_equal(t16^.wait(), expected)


def _as_immutable_ptr(
    arr: Array[Float32, _SCALE_SIZE]
) -> Pointer[Float32, ImmutAnyOrigin]:
    return arr.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()


async def _square_n_times(
    ctx: Context, input: Array[Float32, _SCALE_SIZE], times: Int
) raises -> Array[Float32, _SCALE_SIZE]:
    """Squares `input` element-wise, `times` times in a row, launching a
    real device round-trip (and a real device sync) for each squaring.

    Inlines the copy/launch/copy sequence directly rather than calling
    `square()` in a loop: repeatedly `await`-ing the *same* nested async
    function from within a loop hung indefinitely during development of
    this test on this nightly toolchain, while inlining the identical
    device work did not. Narrowed down to: a raising or non-raising nested
    coroutine awaited once is fine (see `test_raising_task.mojo`'s
    nested-coroutine test); awaiting the same nested async function
    repeatedly from a loop, specifically when it does real GPU host-API
    work, is what triggers it. Worth its own follow-up issue against the
    nightly toolchain rather than working around it silently here.
    """
    var current = input.copy()
    for i in range(times):
        var device_buffer = ctx.gpu_ctx().enqueue_create_buffer[DType.float32](
            _SCALE_SIZE
        )
        _copy_to_device[_SCALE_SIZE](
            ctx, device_buffer, _as_immutable_ptr(current)
        )
        await ctx.synchronize()

        _launch_square_kernel[_SCALE_SIZE](ctx, device_buffer)

        var result = Array[Float32, _SCALE_SIZE](uninitialized=True)
        _copy_from_device[_SCALE_SIZE](
            ctx,
            result.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            device_buffer,
        )
        await ctx.synchronize()
        current = result^
    return current^


def _expected_after_squaring(base: Float32, times: Int) -> Float32:
    var value = base
    var i = 0
    while i < times:
        value *= value
        i += 1
    return value


def _spawn_and_verify_non_uniform_yields[
    N: Int
](mut executor: Executor, context: Context) raises:
    comptime if N == 0:
        executor.wait()
    else:
        # Every task launches its own real GPU kernel work, but a different
        # number of times -- 0 through 3 -- so the executor is coalescing
        # syncs across tasks with non-uniform yield counts, not just a
        # uniform round of same-shaped work.
        var times = N % 4
        var base = Float32(2)
        var input: Array[Float32, _SCALE_SIZE] = [
            base,
            base,
            base,
            base,
            base,
            base,
            base,
            base,
        ]
        var task = executor.add(_square_n_times(context, input, times))
        _spawn_and_verify_non_uniform_yields[N - 1](executor, context)
        var expected = _expected_after_squaring(base, times)
        var got = task^.wait()
        for i in range(_SCALE_SIZE):
            assert_equal(got[i], expected)


def test_synchronize_handles_non_uniform_yield_counts_across_gpu_tasks() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)
        var context = executor.context()
        _spawn_and_verify_non_uniform_yields[12](executor, context)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
